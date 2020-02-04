"""Layers needed to create the MPNN model.

Taken from the ``tf2`` branch of the ``nfp`` code:

https://github.com/NREL/nfp/blob/tf2/examples/tf2_tests.ipynb
"""

"""Adapted from https://github.com/NREL/nfp/blob/tf2/examples/tf2_tests.ipynb"""

import tensorflow as tf
from tensorflow.keras import layers

if tf.__version__ < '1.15.0':
    from tensorflow.python.ops.ragged.ragged_util import repeat
else:
    repeat = tf.repeat


class MessageBlock(layers.Layer):
    """Message passing layer for MPNNs

    Takes the state of an atom and bond, and updates them by passing messages between nearby neighbors.

    Following the notation of Gilmer et al., the message function summs all of the atom states from
    the neighbors of each atom and then updates the node state by adding them to the previous state.
    """

    def __init__(self, atom_dimension, **kwargs):
        """
        Args:
             atom_dimension (int): Number of features to use to describe each atom
        """
        super(MessageBlock, self).__init__(**kwargs)
        self.atom_bn = layers.BatchNormalization()
        self.bond_bn = layers.BatchNormalization()
        self.bond_update_1 = layers.Dense(2 * atom_dimension, activation='softmax', use_bias=False)
        self.bond_update_2 = layers.Dense(atom_dimension)
        self.atom_update = layers.Dense(atom_dimension, activation='softmax', use_bias=False)
        self.atom_dimension = atom_dimension

    def call(self, inputs):
        original_atom_state, original_bond_state, connectivity = inputs

        # Batch norm on incoming layers
        atom_state = self.atom_bn(original_atom_state)
        bond_state = self.bond_bn(original_bond_state)

        # Gather atoms to bond dimension
        target_atom = tf.gather(atom_state, connectivity[:, 0])
        source_atom = tf.gather(atom_state, connectivity[:, 1])

        # Update bond states with source and target atom info
        new_bond_state = tf.concat([source_atom, target_atom, bond_state], 1)
        new_bond_state = self.bond_update_1(new_bond_state)
        new_bond_state = self.bond_update_2(new_bond_state)

        # Update atom states with neighboring bonds
        source_atom = self.atom_update(source_atom)
        messages = source_atom * new_bond_state
        messages = tf.math.segment_sum(messages, connectivity[:, 0])

        # Add new states to their incoming values (residual connection)
        bond_state = original_bond_state + new_bond_state
        atom_state = original_atom_state + messages

        return atom_state, bond_state
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_dimension': self.atom_dimension
        })
        return config


class GraphNetwork(layers.Layer):
    """Layer that implements an entire MPNN neural network

    Create shte message passing layers and also implements reducing the features of all nodes in
    a graph to a single feature vector for a molecule.

    The reduction to a single feature for an entrie molecule is produced by summing a single scalar value
    used to represent each atom. We chose this reduction approach under the assumption the energy of a molecule
    can be computed as a sum over atomic energies."""

    def __init__(self, atom_classes, bond_classes, atom_dimension, num_messages, **kwargs):
        """
        Args:
             atom_classes (int): Number of possible types of nodes
             bond_classes (int): Number of possible types of edges
             atom_dimension (int): Number of features used to represent a node and bond
             num_messages (int): Number of message passing steps to perform
        """
        super(GraphNetwork, self).__init__(**kwargs)
        self.atom_embedding = layers.Embedding(atom_classes, atom_dimension, name='atom_embedding')
        self.atom_mean = layers.Embedding(atom_classes, 1, name='atom_mean')
        self.bond_embedding = layers.Embedding(bond_classes, atom_dimension, name='bond_embedding')
        self.message_layers = [MessageBlock(atom_dimension) for _ in range(num_messages)]
        self.output_atomwise_dense = layers.Dense(1)
        self.dropout_layer = layers.Dropout(.5)

    def call(self, inputs):
        atom_types, bond_types, node_graph_indices, connectivity = inputs

        # Initialize the atom and bond embedding vectors
        atom_state = self.atom_embedding(atom_types)
        bond_state = self.bond_embedding(bond_types)

        # Perform the message passing
        for message_layer in self.message_layers:
            atom_state, bond_state = message_layer([atom_state, bond_state, connectivity])

        # Add some dropout before hte last year
        atom_state = self.dropout_layer(atom_state)

        # Reduce atom to a single prediction
        atom_solubility = self.output_atomwise_dense(atom_state) + self.atom_mean(atom_types)

        # Sum over all atoms in a mol
        mol_energy = tf.math.segment_sum(atom_solubility, node_graph_indices)

        return mol_energy

    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_classes': self.atom_embedding.input_dim,
            'bond_classes': self.bond_embedding.input_dim,
            'atom_dimension': self.atom_embedding.output_dim,
            'num_messages': len(self.message_layers)
        })
        return config


class Squeeze(layers.Layer):
    """Wrapper over the tf.squeeze operation"""

    def __init__(self, axis=1, **kwargs):
        """
        Args:
            axis (int): Which axis to squash
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config


custom_objects = {
    'GraphNetwork': GraphNetwork,
    'MessageBlock': MessageBlock,
    'Squeeze': Squeeze
}
