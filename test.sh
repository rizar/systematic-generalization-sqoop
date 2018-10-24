set -e

ARGS="--batch_size=2 --record_loss_every=1 --checkpoint_every=2 --num_iterations=5 --num_val_samples=4"
export NMN=`pwd`

python scripts/generate_sqoop.py --num-shapes=3 --num_repeats=100 --rhs_variety=1 --font=DejaVuSans.ttf
bash scripts/train/film_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $ARGS --checkpoint_path=film_flatqa
bash scripts/train/mac_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $ARGS --checkpoint_path=mac_flatqa
bash scripts/train/ee_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $ARGS --checkpoint_path=ee_flatqa
bash scripts/train/ee_new_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $ARGS --checkpoint_path=ee_new_flatqa
bash scripts/train/rel_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $ARGS --checkpoint_path=rel_flatqa
bash scripts/train/convlstm_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $ARGS --checkpoint_path=convlstm_flatqa
bash scripts/train/shnmn_flatqa.sh --data_dir sqoop-variety_1-repeats_100 $ARGS --checkpoint_path=shnmn_flatqa
