#!/usr/bin/python3

from shutil import rmtree
from os import walk, mkdir
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from models import Llama3, Qwen2
from chains import label_tanl_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing pdfs')
  flags.DEFINE_string('output_dir', default = 'output', help = 'path to output directory')
  flags.DEFINE_enum('model', default = 'qwen2', enum_values = {'llama3', 'qwen2'}, help = 'model options')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  tokenizer, llm = {
    'llama3': Llama3,
    'qwen2': Qwen2
  }[FLAGS.model](locally = True)
  chain = label_tanl_chain(llm, tokenizer)
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext.lower() in ['.htm', '.html']:
        loader = UnstructuredHTMLLoader(join(root, f))
      elif ext.lower() == '.txt':
        loader = TextLoader(join(root, f))
      elif ext.lower() == '.pdf':
        loader = UnstructuredPDFLoader(join(root, f), mode = 'single', strategy = "hi_res")
      else:
        raise Exception('unknown format!')
      text = ' '.join([doc.page_content for doc in loader.load()])
      output = chain.invoke({'text': text})
      with open(join(FLAGS.output_dir, '%s_meta.txt' % splitext(f)[0]), 'w') as fp:
        fp.write(output)

if __name__ == "__main__":
  add_options()
  app.run(main)
