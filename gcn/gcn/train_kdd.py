from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import pickle
from gcn.utils import *
from gcn.models import GCN, MLP, GCN_dense
import datetime
import os

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'kdd', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string("trained_our_path", '200502160359', "The path for the trained model")

if_save_model = True
restore_trained_our = False
if_save_hidden = True
checkpoints_dir_base = "./checkpoints"
current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
checkpoints_dir = os.path.join(checkpoints_dir_base, current_time, current_time)
# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_kdd(FLAGS.dataset)
#adj_cora, features_cora, y_train_cora, y_val_cora, y_test_cora, train_mask_cora, val_mask_cora, test_mask_cora = load_data("cora")
# Some preprocessing
#features = preprocess_features_kdd_simple(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features.shape[1], logging=True)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
# Initial the training process
var_list = tf.global_variables()
saver = tf.train.Saver(var_list, max_to_keep=10)
if if_save_model:
    os.mkdir(os.path.join(checkpoints_dir_base, current_time))
    saver.save(sess, checkpoints_dir)  # save the graph

if restore_trained_our:
    #checkpoints_dir_our = "./checkpoints"
    checkpoints_dir_our = os.path.join(checkpoints_dir_base, FLAGS.trained_our_path)
    saver.restore(sess,tf.train.latest_checkpoint(checkpoints_dir_our))
    print("model_load_successfully")
    # reset the opitmizer of the model
    model.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    # save the checkpoint
    if (epoch % 50 == 0) and if_save_model:
        saver.save(sess, checkpoints_dir, global_step=epoch, write_meta_graph=False)
        print("Save the model at epoch:", '%04d' % (epoch + 1))
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break
if if_save_model:
    saver.save(sess, checkpoints_dir, global_step=FLAGS.epochs, write_meta_graph=False)
print("Optimization Finished!")
# save the hidden vectors
if if_save_hidden:
    outs = sess.run([model.activations[-2], model.outputs], feed_dict=feed_dict)
    with open(os.path.join("./hidden_vectors", str(FLAGS.hidden2) +"_" + current_time + ".pkl"), "wb") as f:
        pickle.dump(outs, f)

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
