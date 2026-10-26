#!/usr/bin/env python3
"""Authenticated, one-window-at-a-time 22-autosome context iterator.

This is preparation only.  It never exports a whole cohort tensor, admits a
training run, chooses tiles/radii, or accepts EVALUATE.  Tiles are supplied by
an authenticated request and each yielded view retains separate core, valid and
padding masks from ``EventChromosome.load_window``.
"""
import hashlib,json,re
from pathlib import Path
import numpy as np
import r02_genomic_event_inputs as events

SCHEMA="r02_context_autosomal_dataset_request_v1"
PERSON_BATCH_SCHEMA="r02_context_autosomal_person_batches_v1"
PERSON_PARTITION_SCHEMA="r02_context_person_partition_v1"
READER_BACKEND_SCHEMA="r02_context_reader_backend_v1"
READER_BACKEND_LIMIT_SCHEMA="r02_context_reader_backend_v2"
MMAP_READER_BACKEND="ZIP_STORED_MMAP_V1"
AUTOSOMES=tuple(str(x)for x in range(1,23))
require=events.require

def person_axis_sha256(ids):
    return hashlib.sha256(("\n".join(map(str,ids))+"\n").encode()).hexdigest()

def partition_people(ids,roles,components,max_people):
    """Partition only the storage axis; preserve every frozen person and order."""
    ids,roles,components=tuple(map(str,ids)),tuple(map(str,roles)),tuple(map(str,components))
    require(ids and len(ids)==len(set(ids))and len(roles)==len(components)==len(ids),
            "Ordered person identity/role/component axis differs")
    require(type(max_people)is int and max_people>0,"Positive person batch limit required")
    return tuple((ids[i:i+max_people],roles[i:i+max_people],components[i:i+max_people])
                 for i in range(0,len(ids),max_people))

def _json(path,digest):
    require(re.fullmatch(r"[0-9a-f]{64}",digest or"")and events.sha(path)==digest,"Autosomal request SHA256 differs")
    def unique(pairs):
        out={}
        for key,value in pairs:require(key not in out,"duplicate request key");out[key]=value
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=unique)

class AutosomalDataset:
    """Retain authenticated chromosome metadata; materialize one tile per step."""
    def __init__(self,request_path,request_sha256):
        r=_json(request_path,request_sha256)
        self.request_sha256=request_sha256
        required={"schema","stage","person_ids","roles","component_ids","stores","block_id","exclusion_scope",
                  "common_radius_cm","rare_radius_cm","attention_depth","tiles"}
        require(r.get("schema")in(SCHEMA,PERSON_BATCH_SCHEMA),"Autosomal request schema differs")
        if r["schema"]==PERSON_BATCH_SCHEMA:required.update(("person_partition","reader_backend"))
        require(set(r)==required,"Autosomal request fields differ")
        require(r["stage"]in("FIT","SELECT"),"Only FIT or SELECT autosomal iteration is allowed")
        self.ids=tuple(map(str,r["person_ids"]));self.roles=tuple(map(str,r["roles"]));self.components=tuple(map(str,r["component_ids"]))
        require(self.ids and len(self.ids)==len(set(self.ids))and len(self.roles)==len(self.components)==len(self.ids)
                and set(self.roles)=={r["stage"]}and all(self.components),"Ordered development identities/roles/components differ")
        require(type(r["block_id"])is str and r["block_id"]and type(r["exclusion_scope"])is str and r["exclusion_scope"],"Explicit block/exclusion required")
        require(type(r["common_radius_cm"])in(float,int)and np.isfinite(r["common_radius_cm"])and r["common_radius_cm"]>0,"Explicit common radius required")
        require(type(r["attention_depth"])is int and r["attention_depth"]>=0 and
                ((r["rare_radius_cm"]is None and r["attention_depth"]==0)or
                 (type(r["rare_radius_cm"])in(float,int)and np.isfinite(r["rare_radius_cm"])and r["rare_radius_cm"]>0 and r["attention_depth"]>0)),
                "Explicit rare radius/depth disagree")
        if r["schema"]==PERSON_BATCH_SCHEMA:
            p=r["person_partition"]
            require(set(p)=={"schema","max_people","method","complete_person_axis_sha256"}and
                    p["schema"]==PERSON_PARTITION_SCHEMA and p["method"]=="CONTIGUOUS_FROZEN_ORDER"and
                    p["complete_person_axis_sha256"]==person_axis_sha256(self.ids),
                    "Person partition lineage/method differs")
            self.person_batches=partition_people(self.ids,self.roles,self.components,p["max_people"])
            backend=r["reader_backend"]
            if backend.get("schema")==READER_BACKEND_SCHEMA:
                require(set(backend)=={"schema","name","max_verified_chunks"},
                        "Legacy reader backend fields differ")
                self.max_window_sites=None
            else:
                require(set(backend)=={"schema","name","max_verified_chunks","max_window_sites",
                                      "reader_limits_sha256","context_sha256"}and
                        backend.get("schema")==READER_BACKEND_LIMIT_SCHEMA and
                        type(backend.get("max_window_sites"))is int and backend["max_window_sites"]>=100000 and
                        re.fullmatch(r"[0-9a-f]{64}",backend.get("reader_limits_sha256","") or "")and
                        re.fullmatch(r"[0-9a-f]{64}",backend.get("context_sha256","") or ""),
                        "Versioned external reader row limit differs")
                self.max_window_sites=backend["max_window_sites"]
            require(backend["name"]==MMAP_READER_BACKEND and
                    type(backend["max_verified_chunks"])is int and backend["max_verified_chunks"]>0,
                    "Explicit v2 reader backend/bound differs")
        else:self.person_batches=((self.ids,self.roles,self.components),)
        require(isinstance(r["stores"],list)and len(r["stores"])==22,"Exactly 22 store descriptors required")
        stores={}
        for d in r["stores"]:
            require(set(d)=={"chromosome","path","manifest_sha256"}and str(d["chromosome"])in AUTOSOMES
                    and Path(d["path"]).is_absolute()and re.fullmatch(r"[0-9a-f]{64}",d["manifest_sha256"]),"Invalid store descriptor")
            require(str(d["chromosome"])not in stores,"Duplicate chromosome store");stores[str(d["chromosome"])]=d
        require(tuple(sorted(stores,key=int))==AUTOSOMES and set(r["tiles"])==set(AUTOSOMES),"Exact autosome store/tile inventory required")
        lineage=None;total_chunks=0;authenticated_readers={};metadata_signatures={}
        if r["schema"]==PERSON_BATCH_SCHEMA:
            from r02_event_mmap_reader import authenticate_event_chromosome
        for chrom in AUTOSOMES:
            if r["schema"]==PERSON_BATCH_SCHEMA:
                reader,metadata_signatures[chrom]=authenticate_event_chromosome(
                    stores[chrom]["path"],stores[chrom]["manifest_sha256"])
            else:
                reader=events.EventChromosome(stores[chrom]["path"],stores[chrom]["manifest_sha256"])
            authenticated_readers[chrom]=reader
            require(reader.chrom==chrom,"Store chromosome differs")
            require(r["exclusion_scope"] in {reader.source["exclusion_scope"],*reader.source.get("additional_exclusion_scopes",[])}
                    and sum(x["block_id"]==r["block_id"] for x in reader.blocks)==1,
                    "Block or exclusion scope is not uniformly approved")
            require(all(len(batch[0])<=reader.r["max_batch_people"]for batch in self.person_batches),
                    "Person partition exceeds authenticated store budget")
            actual=[reader.persons[reader.people[x]] if x in reader.people else None for x in self.ids]
            require(all(x is not None and x["role"]==role and x["component"]==component
                        for x,role,component in zip(actual,self.roles,self.components)),"Person role/component differs across stores")
            current=(reader.manifest["event_contract_sha256"],reader.manifest["source_contract_sha256"])
            require(lineage is None or current==lineage,"Autosomal store contracts differ");lineage=current
            total_chunks+=sum(len(reader.manifest["channels"][kind])for kind in("common","rare"))
            tiles=r["tiles"][chrom];require(isinstance(tiles,list)and tiles,"Nonempty supplied tiles required")
            previous=None
            for tile in tiles:
                require(type(tile)is list and len(tile)==2 and all(type(x)is int for x in tile)and tile[0]<=tile[1]
                        and(previous is None or tile[0]==previous+1),"Tiles must be ordered, contiguous and disjoint")
                previous=tile[1]
            require(tiles[0][0]==reader.points[0][0]and tiles[-1][1]==reader.points[-1][0],"Tiles must cover exact mapped first/last bp")
        self.chunk_cache=None
        if r["schema"]==PERSON_BATCH_SCHEMA:
            require(r["reader_backend"]["max_verified_chunks"]==total_chunks,
                    "Reader cache bound must equal the authenticated 22-store chunk inventory")
            from r02_event_mmap_reader import MappedEventChromosome,VerifiedChunkCache
            self.chunk_cache=VerifiedChunkCache(total_chunks)
            self.readers={chrom:MappedEventChromosome.from_authenticated(
                authenticated_readers[chrom],cache=self.chunk_cache,
                metadata_signatures=metadata_signatures[chrom],
                max_window_sites=self.max_window_sites) for chrom in AUTOSOMES}
        else:self.readers=None
        self.request=r;self.stores=stores;self.lineage=lineage;self.training_allowed=False
        self.n_person_batches=len(self.person_batches)
        self.tiles_per_person_batch=sum(len(r["tiles"][chrom]) for chrom in AUTOSOMES)

    def _reader(self,chrom):
        descriptor=self.stores[chrom]
        if self.request["schema"]==PERSON_BATCH_SCHEMA:
            return self.readers[chrom]
        return events.EventChromosome(descriptor["path"],descriptor["manifest_sha256"])

    def iter_person_batch(self,batch_index,start_tile_index=0):
        require(type(batch_index)is int and 0<=batch_index<self.n_person_batches,"Unknown person batch")
        require(type(start_tile_index)is int and 0<=start_tile_index<=self.tiles_per_person_batch,
                "Unknown flat autosomal tile cursor")
        r=self.request
        ids,roles,components=self.person_batches[batch_index]
        flat_index=0
        for chrom in AUTOSOMES:
            reader=self._reader(chrom)
            tiles=r["tiles"][chrom]
            for index,(start,end) in enumerate(tiles):
                if flat_index<start_tile_index:
                    flat_index+=1
                    continue
                view=reader.load_window(list(ids),start,end,r["block_id"],r["exclusion_scope"],
                    r["common_radius_cm"],r["rare_radius_cm"],r["attention_depth"])
                require(tuple(map(str,view["person_id"]))==ids and tuple(map(str,view["role"]))==roles and
                        tuple(map(str,view["component_id"]))==components,"Yielded person identity/order differs")
                for kind in ("common","events"):
                    valid,core,padding=view[kind]["valid"],view[kind+"_core"],view[kind+"_padding"]
                    pos=view[kind+"_pos_bp"];require(valid.dtype==core.dtype==padding.dtype==np.bool_ and valid.shape==core.shape==padding.shape,
                        "core/valid/padding mask differs")
                    require(not np.any(padding&(valid|core))and not np.any(core&((pos<start)|(pos>end))),"Padding/core escaped supplied tile")
                    if view["excluded_bp"]is not None:
                        lo,hi=view["excluded_bp"];require(not np.any((~padding)&(pos>=lo)&(pos<=hi)),"Excluded coordinate entered tile or halo")
                view["autosomal_tile_index"]=index;view["autosomal_tile_count"]=len(tiles)
                view["autosomal_last_tile"]=index+1==len(tiles);view["training_allowed"]=False
                view["person_batch_index"]=batch_index;view["person_batch_count"]=self.n_person_batches
                view["autosomal_flat_tile_index"]=flat_index
                flat_index+=1
                yield view

    def __iter__(self):
        for batch_index in range(self.n_person_batches):
            yield from self.iter_person_batch(batch_index)

def open_autosomal(request_path,request_sha256):return AutosomalDataset(request_path,request_sha256)
