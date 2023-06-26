import pandas as pd

train_path = './data1.txt'
test_path = './test1.txt'
train_output = './train.txt'
test_output = './test.txt'

# Read train data from train_path
data_df = pd.read_table(train_path, sep='\s+', header=None, names=['user_id', 'item_id', '-'], index_col=None)

print('t')
# Group item IDs by user ID in data_df
grouped_data = data_df.groupby('user_id')['item_id'].apply(list).reset_index()

print('t')
# Convert item IDs to integers and unpack them
# grouped_data['item_id'] = grouped_data['item_id'].apply(lambda x: ' '.join(map(str, [int(item) for item in x])))
grouped_data['item_id'] = grouped_data['item_id'].apply(lambda x: ' '.join(map(str, [str(int(item)) for item in x])))

print('t')

# Save the modified train data to train_path
grouped_data.to_csv(train_output, sep=' ', index=False, header=False)

# Read test data from test_path
test_df = pd.read_table(test_path, sep='\s+', header=None, names=['user_id', 'item_id'])

# Group item IDs by user ID in test_df
grouped_test = test_df.groupby('user_id')['item_id'].apply(list).reset_index()

# Convert item IDs to integers and unpack them
grouped_test['item_id'] = grouped_test['item_id'].apply(lambda x: ' '.join(map(str, [int(item) for item in x])))

# Save the modified test data to test_path
grouped_test.to_csv(test_output, sep=' ', index=False, header=False)


# # Read train.txt and test.txt files
# train_df = pd.read_table(train_path, sep='\s+', header=None, names=['user_id', 'item_id', ' '])
#
# #----------------------
# # Group item IDs by user ID in data_df
# grouped_data = train_df.groupby('user_id')['item_id'].apply(list).reset_index()
# # Convert item IDs to integers and unpack them
# grouped_data['item_id'] = grouped_data['item_id'].apply(lambda x: ' '.join(map(str, [str(int(item)) for item in x])))
# #----------------------
#
# test_df = pd.read_table(test_path, sep='\s+', header=None, names=['user_id', 'item_id', ' '])
#
# # Extract unique user and item IDs from train.txt
# unique_user_ids = train_df['user_id'].unique()
# unique_item_ids = train_df['item_id'].unique()
#
# test_df = test_df[(test_df['user_id'].isin(unique_user_ids)) & (test_df['item_id'].isin(unique_item_ids))]
#
# # Filter test data based on user and item IDs from train.txt
# grouped_test_filtered = test_df.groupby('user_id')['item_id'].apply(lambda x: ' '.join([str(item) for item in x if item in unique_item_ids])).reset_index()
#
# grouped_test = test_df.groupby('user_id')['item_id'].apply(list).reset_index()
#
# # Convert item IDs to integers and unpack them
# grouped_test['item_id'] = grouped_test['item_id'].apply(lambda x: ' '.join(map(str, [int(item) for item in x])))
#
#
# ## Saving
# # Save the modified train data to train.txt
# grouped_data.to_csv(train_output, sep=' ', index=False, header=False)
# # Save the filtered test data to test.txt
# grouped_test_filtered.to_csv(test_output, sep=' ', index=False, header=False)