set -e

FULL_TEST_ARGS="--batch_size=2 --record_loss_every=1 --checkpoint_every=2 --num_iterations=5 --num_val_samples=4 --loader_num_workers=0 --load_features"
QUICK_TEST_ARGS="--batch_size=2 --record_loss_every=1 --num_iterations=1 --num_val_samples=4 --loader_num_workers=0 --load_features"
export NMN=`pwd`

python scripts/generate_sqoop.py --num-shapes=3 --num_repeats=100 --rhs_variety=1 --font=DejaVuSans.ttf

# full tests
bash scripts/train/film_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $FULL_TEST_ARGS --checkpoint_path=film_flatqa
bash scripts/train/mac_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $FULL_TEST_ARGS --checkpoint_path=mac_flatqa
bash scripts/train/ee_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $FULL_TEST_ARGS --checkpoint_path=ee_flatqa
bash scripts/train/ee_new_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $FULL_TEST_ARGS --checkpoint_path=ee_new_flatqa
bash scripts/train/rel_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $FULL_TEST_ARGS --checkpoint_path=rel_flatqa
bash scripts/train/convlstm_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $FULL_TEST_ARGS --checkpoint_path=convlstm_flatqa
bash scripts/train/shnmn_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $FULL_TEST_ARGS --checkpoint_path=shnmn_flatqa

# extra quick tests
bash scripts/train/ee_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $QUICK_TEST_ARGS --checkpoint_path=ee_new_flatqa --nmn_use_film=1
bash scripts/train/ee_new_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $QUICK_TEST_ARGS --checkpoint_path=ee_new_flatqa --nmn_use_film=1
