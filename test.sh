python scripts/dataset/generate_flatqa_letters_2.py --num-shapes=3 --num_repeats=4 --rhs_variety=1 --font=DejaVuSans.ttf
bash scripts/train/film_flatqa.sh --data_dir flatqa-letters-variety_1-repeats_4 --record_loss_every=1 --checkpoint_every=3 --num_iterations=8 --checkpoint_path=film_flatqa
