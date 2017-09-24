# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

unk_threshold=${@}
data_dir="data"
if [ ! -d "$data_dir/human_preprocessed" ]; then mkdir "$data_dir/human_preprocessed"; fi

python scripts/preprocess_questions.py \
  --input_questions_json "$data_dir/CLEVR-Humans/CLEVR-Humans-train.json" \
  --input_vocab_json "$data_dir/vocab.json" \
  --output_h5_file "$data_dir/human_preprocessed/train_human_questions_ut$unk_threshold.h5" \
  --output_vocab_json "$data_dir/human_preprocessed/human_vocab_ut$unk_threshold.json" \
  --expand_vocab 1 \
  --unk_threshold $unk_threshold \
  --encode_unk 1 \

python scripts/preprocess_questions.py \
  --input_questions_json "$data_dir/CLEVR-Humans/CLEVR-Humans-val.json" \
  --input_vocab_json "$data_dir/human_preprocessed/human_vocab_ut$unk_threshold.json" \
  --output_h5_file "$data_dir/human_preprocessed/val_human_questions_ut$unk_threshold.h5" \
  --encode_unk 1

python scripts/preprocess_questions.py \
  --input_questions_json "$data_dir/CLEVR-Humans/CLEVR-Humans-test.json" \
  --input_vocab_json "$data_dir/human_preprocessed/human_vocab_ut$unk_threshold.json" \
  --output_h5_file "$data_dir/human_preprocessed/test_human_questions_ut$unk_threshold.h5" \
  --encode_unk 1
