import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm

def get_named_entities(ner):
    entities = []
    entity_desc = []
    try:
        for entity in ner:
            if entity['entity'].startswith('B'):
                if len(entity_desc) != 0:
                    entities.append(" ".join(entity_desc))
                entity_desc = [entity['word']]
            else: # starts with I
                if entity['word'].startswith('##'):
                    if len(entity_desc) > 0:
                        entity_desc[-1] += entity['word'][2:]
                    else:
                        entity_desc.append(entity['word'][2:])
                else:
                    entity_desc.append(entity['word'])
        if len(entity_desc) != 0:
                    entities.append(" ".join(entity_desc))
    except:
        print(ner, entities, entity_desc)
        raise ValueError(f'Error')
    return entities


def get_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    NER = pipeline("ner", model=model, tokenizer=tokenizer, device='cuda:1')
    return NER

def get_tags(claims):
    def data():
        for claim in claims:
            yield claim
    named_entities = []
    with tqdm(total=len(claims), leave=False, desc='Performing NER') as pbar:
        for output in NER(data()):
            named_entities.append(
                output
            )
            pbar.update(1)
    tags = [get_named_entities(entities_per_claim) for entities_per_claim in named_entities]
    return tags

if __name__ == "__main__":
    NER = get_ner_pipeline()

    paths = {
         'train': 'data/fever/train.jsonl',
        #  'val': 'data/fever/shared_task_dev.jsonl',
        #  'test': 'data/fever/shared_task_test.jsonl'
    }

    for type in paths:
        data = pd.read_json(paths[type], lines=True)
        claims = data['claim'].tolist()
        tags = get_tags(claims)
        data['tags'] = tags
        data.to_json(os.path.join('./data/fever_ner/', os.path.basename(paths[type])), orient='records', lines=True)
        print(f'Processed {paths[type]}')
