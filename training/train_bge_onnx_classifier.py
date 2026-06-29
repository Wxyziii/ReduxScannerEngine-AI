from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_ID="BAAI/bge-small-en-v1.5"
MODEL_REVISION="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def read_jsonl(path:Path)->list[dict]:return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed(rows:list[dict],tokenizer,model,batch_size:int=32):
    import numpy as np
    import torch
    vectors=[]
    model.eval()
    for start in range(0,len(rows),batch_size):
        batch=[row["prompt"] for row in rows[start:start+batch_size]]
        tokens=tokenizer(batch,padding=True,truncation=True,max_length=128,return_tensors="pt")
        with torch.no_grad():output=model(**tokens).last_hidden_state[:,0]
        output=torch.nn.functional.normalize(output,p=2,dim=1);vectors.append(output.cpu().numpy())
    return np.concatenate(vectors)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--data-dir",type=Path,default=Path("data/generated"));parser.add_argument("--out-dir",type=Path,default=Path("dist/models-v1.1.0"));parser.add_argument("--report",type=Path,default=Path("data/evaluations/classifier_evaluation.json"));args=parser.parse_args()
    from transformers import AutoModel,AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    train=read_jsonl(args.data_dir/"router_train.jsonl");validation=read_jsonl(args.data_dir/"router_validation.jsonl");test=read_jsonl(args.data_dir/"router_test.jsonl")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION);encoder=AutoModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    x_train=embed(train,tokenizer,encoder);x_validation=embed(validation,tokenizer,encoder);x_test=embed(test,tokenizer,encoder)
    y_train=[row["label"] for row in train];y_validation=[row["label"] for row in validation];y_test=[row["label"] for row in test]
    classifier=LogisticRegression(max_iter=2000,C=8.0,random_state=20260629).fit(x_train,y_train)
    validation_predictions=classifier.predict(x_validation);test_predictions=classifier.predict(x_test)
    labels=sorted(classifier.classes_.tolist());onnx_model=convert_sklearn(classifier,initial_types=[("embedding",FloatTensorType([None,x_train.shape[1]]))],target_opset=17,options={id(classifier):{"zipmap":False}})
    args.out_dir.mkdir(parents=True,exist_ok=True);model_path=args.out_dir/"redux-router-head-v1.onnx";model_path.write_bytes(onnx_model.SerializeToString());labels_path=args.out_dir/"redux-router-labels-v1.json";labels_path.write_text(json.dumps(labels,indent=2),encoding="utf-8")
    dataset_hash=hashlib.sha256(b"".join((args.data_dir/name).read_bytes() for name in ["router_train.jsonl","router_validation.jsonl","router_test.jsonl"])).hexdigest()
    report={"schemaVersion":"redux-maker.classifier-evaluation.v1","baseModel":MODEL_ID,"baseModelRevision":MODEL_REVISION,"baseModelLicense":"MIT","classifier":"multinomial logistic regression ONNX head","featureCount":int(x_train.shape[1]),"datasetSha256":dataset_hash,"splitCounts":{"train":len(train),"validation":len(validation),"test":len(test)},"validationAccuracy":accuracy_score([row["label"] for row in validation],validation_predictions),"testAccuracy":accuracy_score(y_test,test_predictions),"labels":labels,"testClassificationReport":classification_report(y_test,test_predictions,labels=labels,output_dict=True,zero_division=0),"testConfusionMatrix":confusion_matrix(y_test,test_predictions,labels=labels).tolist(),"modelSha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),"labelsSha256":hashlib.sha256(labels_path.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    if report["validationAccuracy"]<.9 or report["testAccuracy"]<.9:raise SystemExit("classifier evaluation below v1 quality gate")
    print(json.dumps({key:report[key] for key in ["validationAccuracy","testAccuracy","modelSha256"]},indent=2))


if __name__=="__main__":main()
