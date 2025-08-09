python classify_seq.py \
--input_file augmented_filtered_data.npy \
--column_names column_names.txt \
--model_path best_model.pth \
--selected_columns Heading Airspeed Altitude \
--lat_col Latitude \
--lon_col Longitude --lstm_n 20 --dense_n 20 --num_heads 20
