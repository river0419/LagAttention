#export CUDA_VISIBLE_DEVICES=0

model=LagAttention

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
patience=10
daytime=48
gapdis=36

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --d_model 8\
  --d_ff 16\
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --var_num 862 \
  --des 'test' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 6 \
  --train_epochs 20\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --d_model 8\
  --d_ff 16\
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --var_num 862 \
  --des 'test' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 6 \
  --train_epochs 20\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --d_model 8\
  --d_ff 16\
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --var_num 862 \
  --des 'test' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 6 \
  --train_epochs 20\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --d_model 8\
  --d_ff 16\
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --var_num 862 \
  --des 'test' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 6 \
  --train_epochs 20\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window