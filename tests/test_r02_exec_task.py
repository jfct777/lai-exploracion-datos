import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('r02_exec_task',Path(__file__).parents[1]/'bin/r02_exec_task.py')
task=importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class TaskContractTests(unittest.TestCase):
    def test_new_m14_diagnostics_is_an_explicitly_allowed_frozen_tool(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            tool=p/'r02_m14_configuration_diagnostics.py'
            tool.write_text('print("synthetic diagnostics only")\n')
            command=p/'command.json'
            command.write_text(json.dumps(dict(command=['python3',str(tool)],tool_sha256=task.sha(tool))))
            task.run(command,p,p/'receipt.json')
            self.assertEqual(json.loads((p/'receipt.json').read_text())['tool_sha256'],task.sha(tool))

    def test_executes_frozen_allowed_tool_without_shell(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            tool=p/'r02_genomic_pair_evidence.py'
            tool.write_text('print("synthetic tool only")\n')
            command=p/'command.json'
            command.write_text(json.dumps(dict(command=['python3',str(tool)],tool_sha256=task.sha(tool))))
            task.run(command,p,p/'receipt.json')
            self.assertEqual(json.loads((p/'receipt.json').read_text())['command'],['python3',str(tool)])
            with self.assertRaisesRegex(ValueError,'exists'): task.run(command,p,p/'receipt.json')

    def test_rejects_shell_and_unknown_or_changed_code(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            f=p/'r02_common_grm.sh'
            f.write_text('exit 0\n')
            command=p/'command.json'
            for args,digest in [(['bash','-c','echo unsafe'],task.sha(f)),
                                 (['bash',str(f)],'0'*64)]:
                command.write_text(json.dumps(dict(command=args,tool_sha256=digest)))
                with self.assertRaises(ValueError): task.run(command,p,p/'receipt.json')


if __name__=='__main__': unittest.main()
