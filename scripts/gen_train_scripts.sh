cd 2k_ta2en
python scripts/create_nmt_train_script.py -p bpe_2k_ta2en -q qa-xp-009 -i 0 -w . -n baseline_ta
qsub train.job
cd ..

cd 2k_ur2en
python scripts/create_nmt_train_script.py -p bpe_2k_ur2en -q qa-xp-009 -i 1 -w . -n baseline_ur
qsub train.job
cd ..

cd 12k_ha2en
python scripts/create_nmt_train_script.py -p bpe_12k_ha2en -q qa-xp-009 -i 2 -w . -n baseline_ha
qsub train.job
cd ..

cd 12k_hu2en
python scripts/create_nmt_train_script.py -p bpe_12k_hu2en -q qa-xp-009 -i 3 -w . -n baseline_hu
qsub train.job
cd ..

cd 12k_tu2en
python scripts/create_nmt_train_script.py -p bpe_12k_tu2en -q qa-xp-010 -i 1 -w . -n baseline_tu
qsub train.job
cd ..

cd 12k_uz2en
python scripts/create_nmt_train_script.py -p bpe_12k_uz2en -q qa-xp-010 -i 2 -w . -n baseline_uz
qsub train.job
cd ..

cd 12k_en2vi
python scripts/create_nmt_train_script.py -p bpe_12k_en2vi -q qa-xp-010 -i 3 -w . -n baseline_vi
qsub train.job
cd ..

