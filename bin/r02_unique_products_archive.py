#!/usr/bin/env python3
"""Archive useful file bytes once, with verified path/link reconstruction metadata.

Factored from the useful stage archive in the LAI comparison operator. The
caller supplies every exclusion and the byte bound; no scientific file suffix
is discarded implicitly. This module never uploads, deletes or retires a VM.
"""
from pathlib import Path, PurePosixPath
import hashlib
import os
import tarfile

import r02_publish_evidence as pub
pipeline=pub.pipeline
require=pub.require


def relative_member(value):
    require(isinstance(value,str)and value and not any(c in value for c in ('\\','\0','\n','\r')),
            'Unsafe archive-relative name')
    path=PurePosixPath(value)
    require(not path.is_absolute()and '..'not in path.parts and str(path)==value and value!='.',
            'Unsafe archive-relative path')
    return value


def verify(archive, inventory):
    """Check the exact member set and every distinct scientific payload."""
    expected={}
    for entry in inventory['entries']:
        relative_member(entry['path'])
        if entry['disposition']!='PRESERVE':continue
        member=relative_member(entry['archive_member'])
        identity=(entry['sha256'],entry['bytes'])
        require(member not in expected or expected[member]==identity,'Conflicting duplicate identity in inventory')
        expected[member]=identity
    with tarfile.open(archive,'r:*')as tar:
        members=tar.getmembers()
        require(len(members)==len(expected)and {m.name for m in members}==set(expected),'Archive member inventory differs')
        for member in members:
            require(member.isfile(),'Archive contains an unexpected special member')
            digest,size=expected[member.name]
            require(member.size==size,'Archived payload size differs')
            with tar.extractfile(member)as stream:
                require(hashlib.file_digest(stream,'sha256').hexdigest()==digest,'Archived payload SHA differs')
    return dict(unique_files=len(expected),unique_bytes=sum(v[1]for v in expected.values()))


def build(root,archive,inventory_path,*,exclusions,max_unique_bytes,scope,preferred_prefixes=('published/','results/')):
    """Build a create-only tar/inventory; exclusions are exact paths or prefixes.

    A key ending in '/' excludes that subtree, with its caller-supplied reason.
    Symlinks are recorded before directory checks and never followed. References
    outside root remain explicit; the caller must authenticate their sources.
    """
    root=Path(root);archive=Path(archive);inventory_path=Path(inventory_path)
    require(root.is_absolute()and root.resolve()==root and root.is_dir()and not root.is_symlink(),'Canonical owned root required')
    require(type(max_unique_bytes)is int and max_unique_bytes>0 and scope,'Explicit archive scope/byte bound required')
    require(not archive.exists()and not inventory_path.exists(),'Preserve previous or partial archive')
    require(isinstance(exclusions,dict)and all(isinstance(k,str)and k and not k.startswith('/')and '..'not in Path(k).parts
        and isinstance(v,str)and v for k,v in exclusions.items()),'Explicit safe exclusions with reasons required')
    for destination in [archive,inventory_path]:
        if destination.is_relative_to(root):
            rel=destination.relative_to(root).as_posix()
            require(any((k.endswith('/')and rel.startswith(k))or rel==k for k in exclusions),'Generated archive must be excluded from its own input')
    def reason(rel):
        return next((v for k,v in exclusions.items()if(k.endswith('/')and(rel==k[:-1]or rel.startswith(k)))or rel==k),None)
    def order(path):
        rel=path.relative_to(root).as_posix()
        return (next((i for i,prefix in enumerate(preferred_prefixes)if rel.startswith(prefix)),len(preferred_prefixes)),rel)
    entries=[];seen={};selected=[];total=0
    for path in sorted(root.rglob('*'),key=order):
        rel=path.relative_to(root).as_posix();excluded=reason(rel)
        if excluded:
            entries.append(dict(path=rel,disposition='EXCLUDE',reason=excluded));continue
        if path.is_symlink():
            target=os.readlink(path)
            entries.append(dict(path=rel,disposition='REFERENCE',kind='SYMLINK',target=target,
                target_resolved=str(path.resolve()),target_inside_root=path.resolve().is_relative_to(root),
                reason='Relationship retained without following or copying the staged target'))
            continue
        if path.is_dir():continue
        require(path.is_file(),'Unclassified special file: '+rel)
        before=(path.stat().st_dev,path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
        digest=pipeline.publication_digests(path)
        require(before==(path.stat().st_dev,path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns),'Input changed during inventory: '+rel)
        identity=(digest['sha256'],digest['bytes'])
        if identity not in seen:
            seen[identity]=rel;selected.append((rel,path,digest,before));total+=digest['bytes']
            require(total<=max_unique_bytes,'Useful archive exceeds declared byte allowance')
        entries.append(dict(path=rel,disposition='PRESERVE',archive_member=seen[identity],**digest))
    require(total>0,'Empty useful archive')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'x',format=tarfile.PAX_FORMAT,dereference=False)as tar:
        for rel,path,digest,before in selected:
            require(before==(path.stat().st_dev,path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns),'Input changed before archive: '+rel)
            tar.add(path,arcname=rel,recursive=False)
    value=dict(schema='r02_unique_useful_products_archive_v1',root=str(root),scope=scope,entries=entries,
        unique_bytes=total,duplicate_payloads_removed=True,source_deleted=False,
        nextflow_cache_automatically_restored=False,
        restoration='Restore canonical archive members, recreate exact-byte duplicate paths from archive_member and symlinks from their explicit targets; authenticate any outside-root source separately.')
    verified=verify(archive,value)
    require(verified['unique_bytes']==total,'Archive byte inventory differs')
    # Also guard every duplicate path: equivalence must hold at capture completion.
    for entry in entries:
        if entry['disposition']=='PRESERVE':
            require(pipeline.publication_digests(root/entry['path'])=={k:entry[k]for k in ('sha256','bytes','md5_base64')},
                    'Source changed before snapshot completion: '+entry['path'])
    value['archive_sha256']=pipeline.sha(archive)
    pub.write_receipt_new(inventory_path,value)
    return value
