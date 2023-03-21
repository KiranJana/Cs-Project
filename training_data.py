#!/usr/bin/env python
# coding: utf-8

import os
import json
import pickle
from collections import namedtuple
import re

from labelbox.schema.ontology import OntologyBuilder, Tool, Classification, Option
from labelbox import Client, LabelingFrontend, LabelImport, MALPredictionImport
from labelbox.data.annotation_types import (
    Label,
    TextData,
    TextEntity,
    Checklist,
    Radio,
    ObjectAnnotation,
    TextEntity,
    ClassificationAnnotation,
    ClassificationAnswer,
)
from labelbox.data.serialization import NDJsonConverter
import click
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens import DocBin
import pandas as pd
from google.cloud import storage
import uuid
import concurrent.futures
import time


class Annotation:
    def __init__(self, start: int = None, end: int = None, label: str = None):
        self.start = start
        self.end = end
        self.label = label

    @classmethod
    def from_json(cls, json_str: str = None):
        annot_dict = json.loads(json_str)
        annotation = cls(
            start=annot_dict.get("start"),
            end=annot_dict.get("end"),
            label=annot_dict.get("label"),
        )
        return annotation

    @classmethod
    def from_dict(cls, annot_dict: dict = None):
        annotation = cls(
            start=annot_dict.get("start"),
            end=annot_dict.get("end"),
            label=annot_dict.get("label"),
        )
        return annotation

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return f"Annotation(start={self.start}, end={self.end}, label={self.label})"


class TrainingData:
    """
    Training data for StartDate NLP project.
    Attributes:\n
    data: text with NER labels, format: [(text, [Annotations(start, end, label)])]
    labels: All NER labels, format [str(label)]
    Methods:\n
    pre_label_data:
    from_json:
    from_pickle:
    from_labelbox_data:
    load_from_json:
    to_json:
    to_pickle:
    save_as_json:
    upload_data_to_labelbox:
    """

    def __init__(self, data: list = None, labels: list = None):
        self.data = data
        self.labels = labels

    @staticmethod
    def _create_objects(annotation: Annotation):
        named_enity = TextEntity(start=annotation.start, end=annotation.end-1)     # end must be decreased by 1 as the label in Labelbox does include the upper bound
        named_enity_annotation = ObjectAnnotation(
            value=named_enity, name=annotation.label
        )
        return named_enity_annotation

    @staticmethod
    def _upload_labels(data_row: namedtuple, project, lb_client, ontology_builder):
        if len(data_row.labels) > 0:
            # create labelbox annotation object for data_row from annotations
            label_annotations = [
                TrainingData._create_objects(annot) for annot in (data_row.labels)
            ]
            label_data = data_row.Index

            label = Label(data=TextData(uid=label_data), annotations=label_annotations)
            label.assign_feature_schema_ids(ontology_builder.from_project(project))
            ndjson_labels = list(NDJsonConverter.serialize([label]))
            # upload to labelbox as prediction labels
            upload_job = MALPredictionImport.create_from_objects(
                client=lb_client,
                project_id=project.uid,
                name="upload_label_import_job" + uuid.uuid4().hex,
                predictions=ndjson_labels,
            )

            # report status
            if len(upload_job.errors) > 0:
                print(f"Errors: {upload_job.errors}")

    @staticmethod
    def _check_tokens_edge(label_token_diff: dict):
        min_value = 20000
        key = 0
        for k in label_token_diff.keys():
            if abs(k) < min_value:
                min_value = abs(k)
                key = k
        token_index = label_token_diff[key]
        return token_index

    @staticmethod
    def _prepare_gcs_path(path: str) -> tuple[str]:
        core_path = path[path.find("gs://") + len("gs://") :]
        path_elems = core_path.split("/")
        bucket_name = path_elems.pop(0)
        file_name = "/".join(path_elems)
        return bucket_name, file_name

    @classmethod
    def pre_label_data(
        cls, scraped_data: list, rules_bucket_name: str, rules_blob_name: str
    ) -> list:
        # get prelabeling pipeline json
        client = storage.Client()
        bucket = client.get_bucket(rules_bucket_name)
        blob = bucket.blob(rules_blob_name)
        rules = json.loads(blob.download_as_string())

        # instantiate model
        spacy.require_cpu()
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(rules)

        docs = nlp.pipe(scraped_data)

        # add labels
        spacy_labels = list(nlp.get_pipe("entity_ruler").labels)
        try:
            spacy_labels.extend(list(nlp.get_pipe("ner").labels))
        except KeyError:
            print("Info: no ner component in pipeline")
        spacy_labels.sort()

        # create entities data
        ents_list = [
            (
                doc.text,
                [
                    Annotation(ent.start_char, ent.end_char, ent.label_)
                    for ent in doc.ents
                ],
            )
            for doc in docs
        ]
        return cls(data=ents_list, labels=spacy_labels)

    @classmethod
    def from_json(cls, json_data: str):
        '''
        Creates TrainingData instance from json string.
        Params:\n
        :: json_data: json string
        '''
        deserialized_ents_list = [
            (text, [Annotation.from_dict(annot) for annot in annotations])
            for text, annotations in json.loads(json_data)
        ]
        
        labels = list(
            {
                j
                for i in [[annot.label for annot in item[1]] for item in deserialized_ents_list]
                for j in i
            }
        )
        return cls(data=deserialized_ents_list, labels=labels)

    @classmethod
    def load_from_json_file(cls, load_path: str):
        """
        Convenience method for loading training data instance from json file.
        params:
        :load_path: provide local file system path (relative or abslolute) or GC storage path as "gs://bucketname/blobname"
        """
        if "gs://" in load_path:
            # extract
            bucket_name, file_name = TrainingData._prepare_gcs_path(load_path)

            # download file
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(file_name)
            data_json = blob.download_as_string()

        else:
            # load local file
            with open(load_path, "r") as f:
                data_json = f.read()

        # instantiate with the downloaded json
        return cls.from_json(data_json)

    @classmethod
    def from_pickle(cls, pickle_data: bytes):
        return cls(data=pickle.loads(pickle_data), labels=None)

    @classmethod
    def from_labelbox_data(
        cls, labelbox_project_name: str, labelbox_api_key: str = None
    ):
        """
        Create TrainingData instance from Labelbox downloaded data.
        Params:
        :: labelbox_project_name: project to be downloaded
        :: labelbox_api_key: API key, if not provided, environment variable LABELBOX_API_KEY is looked for
        """

        # connect to labelbox and get project name and id
        if not labelbox_api_key:
            lb_client = Client(os.getenv("LABELBOX_API_KEY"))
        try:
            project_name_and_id = [
                (p.name, p.uid)
                for p in lb_client.get_projects()
                if p.name == labelbox_project_name
            ][0]
        except IndexError:
            raise IndexError("Could not find project, perhaps wrong name.")
        # instantiate project based on project ID
        project = lb_client.get_project(project_name_and_id[1])

        labels = project.label_generator()
        labeled_data = ((l.data.text, l.annotations) for l in labels)

        # create training data
        training_data = [
            (
                text,
                [Annotation(row.value.start, row.value.end+1, row.name) for row in label],    # must add 1 as the labels in labelbox include the upper bound
            )
            for text, label in labeled_data
        ]

        if len(training_data) == 0:
            raise ValueError("Empty dataset.")

        # create labels from loaded training data
        labels = list(
            {
                j
                for i in [[annot.label for annot in item[1]] for item in training_data]
                for j in i
            }
        )
        return cls(data=training_data, labels=labels)


    @classmethod
    def merge(cls, training_data_list: list):
        '''
        Create a new TrainingData instance by merging other TrainingData.
        Removes all duplicates in the merged data.
        Params:\n
        trainig_data_list: list of instances to be merged
        '''
        new_set = set()
        new_labels = set()
        for t in training_data_list:
            # process data attr
            # convert to set to remove duplicates in data
            t_set = set([(text, json.dumps([annot.to_dict() for annot in annotations])) for text, annotations in t.data])
            new_set = new_set.union(t_set)

            # process labels if exist
            if t.labels:
                new_labels = new_labels.union(set(t.labels))
        # convert to list and deserialize
        merged_data = [(text, [Annotation.from_dict(annot) for annot in json.loads(annotations)]) for text, annotations in new_set]       
        merged_labels = list(new_labels)
        return cls(data=merged_data, labels=merged_labels)


    def to_json(self) -> str:
        serialized_ents_list = json.dumps(
            [
                (text, [annot.to_dict() for annot in annotations])
                for text, annotations in self.data
            ]
        )
        return serialized_ents_list

    def to_pickle(self):
        serialized_ents_list = pickle.dumps(self.data)
        return serialized_ents_list

    def save_as_json(self, save_path: str):
        """
        Convenience method for saving training data instance as json file.
        params:
        :save_path: provide local file system path (relative or abslolute) or GC storage path as "gs://bucketname/blobname"
        """

        if "gs://" in save_path:
            # extract
            bucket_name, file_name = TrainingData._prepare_gcs_path(save_path)

            # upload file
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(file_name)
            blob.upload_from_string(self.to_json(), content_type="application/json")
        else:
            # save local file
            with open(save_path, "w") as w:
                w.write(self.to_json())

    def upload_data_to_labelbox(
        self,
        labelbox_project_name: str,
        labelbox_dataset_name: str = None,
        labelbox_api_key: str = None,
    ) -> None:
        """
        Loads TrainingData to Labelbox.
        params:
        :: bucket_name: GCS bucket with scraped data
        :: ruler_pattern_blob_name: blob name with rules for spacy entity ruler
        :: labelbox_project_name: Name of Labelbox project to upload data (will be created if not exists)
        :: labelbox_dataset_name: Name of Labelbox dataset to be created from scraped data
        :: labelbox_api_key_env_var: Name of environment variable holding Labebox API key
        """

        # create namedtuple for better readability
        row_data = namedtuple("row_data", ["text", "labels", "ext_id"])
        # add uuid to the data
        raw_data = [
            row_data(text, labels, uuid.uuid4().hex) for text, labels in self.data
        ]
        # create dataset for labelbox
        data = [{"row_data": t.text, "external_id": t.ext_id} for t in raw_data]

        # instantiate labelbox client
        if not labelbox_api_key:
            labelbox_api_key = os.getenv("LABELBOX_API_KEY")
        lb_client = Client(api_key=labelbox_api_key)
        # Create a new dataset
        if not labelbox_dataset_name:
            labelbox_dataset_name = labelbox_project_name + "_dataset"
        dataset = lb_client.create_dataset(name=labelbox_dataset_name)
        dataset_uid = dataset.uid

        # Bulk add data rows to the dataset
        start_time = time.perf_counter()
        task = dataset.create_data_rows(data)

        task.wait_till_done()
        end_time = time.perf_counter()
        seconds = end_time - start_time
        print(f"{task.status}, total time (s)={round(seconds,2)}")

        # convert to pandas for merge
        raw_data_df = pd.DataFrame(raw_data)
        result_df = pd.DataFrame(task.result)
        # merge and filter out columns
        labeled_data_rows = pd.merge(
            result_df, raw_data_df, left_on="external_id", right_on="ext_id"
        )[["id", "row_data", "labels"]].set_index("id")

        # prepare Tools
        if not self.labels:
            self.labels = list(
                {
                    j
                    for i in [[annot.label for annot in item[1]] for item in self.data]
                    for j in i
                }
            )
        tools = [Tool(tool=Tool.Type.NER, name=label) for label in self.labels]

        # set-up project
        # change project name as appropriate
        project = lb_client.create_project(name=labelbox_project_name)

        ontology_builder = OntologyBuilder(tools=tools)

        editor = next(
            lb_client.get_labeling_frontends(where=LabelingFrontend.name == "Editor")
        )
        # create project
        project.setup(editor, ontology_builder.asdict())
        # connect previously created dataset to project
        project.datasets.connect(dataset)

        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for data_row in labeled_data_rows.itertuples():
                executor.submit(
                    self._upload_labels, data_row, project, lb_client, ontology_builder
                )

        end_time = time.perf_counter()
        seconds = end_time - start_time
        print(
            f"Total time={round(seconds,2)}, Time per item={round(seconds/len(labeled_data_rows.index),2)}"
        )

    def transform_to_spacy(self) -> list[Doc]:
        '''
        Parses training data labels to spacy tokens and fixes instances where the label start/end differs from token start/end.
        Checks if any named entity includes leading or trailing whitespace and removes it.
        Returns list of spacy.token.doc.Doc.
        Hint: to serialize, use spacy.

        '''
        # instansiate vars and dependencies
        nlp = spacy.blank("en")
        docs = []
        errs = []
        err = namedtuple("err", ["text", "start", "end", "label"])
        # iterate over all training data
        for idx, row in enumerate(self.data):
            text = row[0]
            labels = row[1]

            # spacy tokenize text
            doc = nlp.tokenizer(text)
            label_data = []
            # check alignment with tokens
            for label in labels:
                span = doc.char_span(label.start, label.end)
                try:
                    label = spacy.tokens.Span(doc, span.start, span.end, label.label)
                    # label_data.append(label)
                except AttributeError:
                    print(f"Info: idx={idx}, error at parsing {text[label.start: label.end]}, {label.label}, at position: {label.start}, {label.end}; fixing now...") # move to logger
                    errs.append(err(text, label.start, label.end, label.label)) # move to logger
                    label_token_diff_start = {t.idx-label.start: t.idx for t in doc}
                    label_token_diff_end = {(t.idx + len(t))-label.end: t.idx + len(t) for t in doc}
                    if 0 not in label_token_diff_start.keys():
                        # get the closest value
                        start_index = TrainingData._check_tokens_edge(label_token_diff_start)
                        label.start = start_index

                    if 0 not in label_token_diff_end.keys():
                        # get the closest value
                        end_index = TrainingData._check_tokens_edge(label_token_diff_end)
                        label.end = end_index
                    
                    span = doc.char_span(label.start, label.end)
                    label = spacy.tokens.Span(doc, span.start, span.end, label.label)
                    
                label_data.append(label)
            # inject spans into tokenized text, entities
            doc.ents = label_data
            # check leading/trailing whitespace for new entities 
            new_ents = []
            for ent in doc.ents:
                match_ent = tuple(re.finditer("^\W+|\W+$", ent.text)) 
                if len(match_ent) != 0:  
                    start_offset, end_offset = 0, 0
                    for i in match_ent:
                        if i.start() == 0:
                            start_offset = i.end()
                        if i.start() != 0:
                            end_offset = i.end() - i.start()    
                    new_ent = doc.char_span(ent.start_char + start_offset, ent.end_char - end_offset, label=ent.label_)        
                    new_ents.append(new_ent)
                else:
                    new_ents.append(ent)
            doc.ents = new_ents
            docs.append(doc)
        return docs
        
    def transform_to_spacy_and_save_docs(self, save_path: str) -> None:
        '''
        Convenience function for transform the data to spacy and save docs as bytes.
        Params:\n
        :save_path: provide local file system path (relative or abslolute) or GC storage path as "gs://bucketname/blobname"
        '''

        docs = self.transform_to_spacy()
        doc_bin = DocBin(attrs=["ORTH", "ENT_IOB", "ENT_TYPE"])
        for doc in docs:
            doc_bin.add(doc)
        bytes_data = doc_bin.to_bytes()

        if "gs://" in save_path:
            # extract
            bucket_name, file_name = TrainingData._prepare_gcs_path(save_path)

            # upload file
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(file_name)
            blob.upload_from_string(bytes_data, content_type="application/octet-stream")
        else:
            # save local file
            with open(save_path, "wb") as w:
                w.write(bytes_data)
        print(f"Docs binary data saved to location {save_path}")

    def __repr__(self):
        return f"TrainingData(data={self.data}, labels={self.labels})"




if __name__ == "__main__":
    from pipeline_cli import main
    main()
    
    # bucket_name = "knappekl_nlp_project"
    # rules_blob_name = "stardate_ruler_patterns.json"
    # file_type = "application/json"
    # scraped_file_name = "test_parser_data.parquet"
    # sav_file_name = "training_data.startdate"
    # save_path = f"gs://{bucket_name}/{sav_file_name}"

    # scraped_data = pd.read_parquet(f"gs://{bucket_name}/{scraped_file_name}")
    # raw_texts = scraped_data["text"][:10].to_list()

    # training_data = TrainingData.pre_label_data(raw_texts, bucket_name, rules_blob_name)

    # # save training data to GCS
    # training_data.save_as_json("data/training_data.json")
    # labelbox_project_name = "startdate_nlp_cli"
    # training_data = TrainingData.from_labelbox_data(labelbox_project_name)
    # print(training_data)
    # labelbox_project_name = "startdate_nlp_new_test"
    # labelbox_dataset_name = "stardate_new dataset_test"

    # training_data.upload_data_to_labelbox(labelbox_project_name)
    # json_load_path = "data/labeled_data.json"
    # with open(json_load_path, "r") as f:
    #     json_string = f.read()
    # training_data = TrainingData.from_json(json_string)
    # json_load_path = "data/labeled_data.json"
    # training_data = TrainingData.load_from_json(json_load_path)
    # doc_list = training_data.transform_to_spacy()
    # training_data.transform_to_spacy_and_save_docs(save_path="gs://knappekl_nlp_project/doc_data.spacy")
