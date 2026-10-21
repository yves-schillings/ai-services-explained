"""Deterministic control tests of the service examples 03 to 12.
No model server or external service is contacted. The training examples 01, 02 and 13 to 18
are tested in test_training.py."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(prefix):
    p=next((ROOT/'examples').glob(prefix+'_*.py'))
    spec=importlib.util.spec_from_file_location('example_'+prefix,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class Examples(unittest.TestCase):
    def test_schema_accepts_valid_draft(self):
        m=load('04');v={'summary':'Server restarted','next_action':'Check cause'}
        self.assertEqual(m.validate(v),v)
    def test_schema_rejects_extra_field(self):
        m=load('04')
        with self.assertRaises(ValueError):m.validate({'summary':'x','next_action':'y','cause':'invented'})
    def test_schema_rejects_empty_value(self):
        m=load('04')
        with self.assertRaises(ValueError):m.validate({'summary':'','next_action':'y'})
    def test_rag_filters_before_embedding(self):
        m=load('05');seen=[]
        def embed(texts):seen.extend(texts);return [[1,0] for _ in texts]
        result=m.retrieve('alice','restart?',embed)
        self.assertEqual([d['id'] for d in result],['D1'])
        self.assertFalse(any('financial' in text for text in seen))
    def test_rag_no_rights_no_model_call(self):
        m=load('05')
        self.assertEqual(m.retrieve('eve','question',lambda _:self.fail('Must not embed')),[])
    def test_rag_low_match_abstains(self):
        m=load('05');self.assertEqual(m.retrieve('alice','unrelated',lambda _:[[1,0],[0,1]]),[])
    def test_invalid_vectors_are_rejected(self):
        with self.assertRaises(ValueError):load('05').cosine([1],[1,2])
    def test_agent_reads_own_ticket_then_finishes(self):
        m=load('06');actions=iter([{'tool':'read_ticket','ticket_id':'T1'},{'tool':'finish','ticket_id':''}])
        result=m.run_agent(lambda _:next(actions));self.assertEqual(result[-1],'finished')
    def test_agent_rejects_foreign_ticket(self):
        with self.assertRaises(PermissionError):load('06').execute({'tool':'read_ticket','ticket_id':'T1'},'bob')
    def test_agent_rejects_shell(self):
        with self.assertRaises(PermissionError):load('06').execute({'tool':'shell','ticket_id':'T1'},'alice')
    def test_agent_step_budget(self):
        with self.assertRaises(RuntimeError):load('06').run_agent(lambda _:{'tool':'read_ticket','ticket_id':'T1'})
    def test_ocr_uses_argument_array_without_shell(self):
        m=load('07')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fictional image.png';p.write_bytes(b'fixture')
            with patch.object(m.subprocess,'run') as run:
                run.return_value.stdout='Fictional text'
                self.assertEqual(m.extract(p),'Fictional text')
                self.assertEqual(run.call_args.args[0][0],'tesseract')
                self.assertNotIn('shell',run.call_args.kwargs)
    def test_ocr_rejects_language_injection(self):
        m=load('07')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scan.png';p.write_bytes(b'fixture')
            with self.assertRaises(ValueError):m.extract(p,'eng; command')
    def test_multimodal_reads_local_image(self):
        m=load('09')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scan.png';p.write_bytes(b'fictional image bytes')
            self.assertEqual(m.image_message(p)['images'],['ZmljdGlvbmFsIGltYWdlIGJ5dGVz'])
    def test_multimodal_rejects_non_image(self):
        m=load('09')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scan.txt';p.write_text('fixture')
            with self.assertRaises(ValueError):m.image_message(p)
    def test_redaction_removes_obvious_email(self):
        self.assertEqual(load('12').redact('learner@example.org'),'[EMAIL REDACTED]')
    def test_redaction_does_not_claim_to_remove_names(self):
        self.assertEqual(load('12').redact('Alex visited on Tuesday.'),'Alex visited on Tuesday.')
    def test_api_rejects_other_endpoints(self):
        with self.assertRaises(ValueError):load('03').local_api('pull',{})
    def test_live_api_shape_with_mocked_transport(self):
        m=load('03')
        with patch.object(m,'local_api',return_value={'message':{'content':'fixture'}}) as api:
            self.assertEqual(m.chat('local-model',[{'role':'user','content':'hello'}]),'fixture')
            self.assertEqual(api.call_args.args[0],'chat')
            self.assertFalse(api.call_args.args[1]['stream'])

if __name__=='__main__':unittest.main(verbosity=2)
