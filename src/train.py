from tensorflow import keras
import tensorflow as tf
import numpy as np
import time


def train(model, train_dataset, test_dataset, epochs, batch_size, pgd_steps, hyperrectangles, hyperrectangles_labels, alpha=1, beta=0, eps_multiplier=1, from_logits=False, optimizer=keras.optimizers.legacy.Adam()):    
    optimizer = optimizer
    ce_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_attack_single_point_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    pgd_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    test_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for x_batch_train, y_batch_train in train_dataset:
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train, training=True) # Outputs for this minibatch
                ce_loss_value = ce_batch_loss(y_batch_train, outputs) # Calculate loss
                ce_loss_value = ce_loss_value * alpha # Multiply by 'alpha'
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(ce_loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        #########################################PGD####################################################
        # Doing the PGD training loop only if 'beta' is > 0. Otherwise the training will be standard.
        if beta:
            pgd_dataset = []
            pgd_labels = []
            for i, hyperrectangle in enumerate(hyperrectangles):
                t_hyperrectangle = np.transpose(hyperrectangle)

                # Calculate the epsilon for each dimension as ((dim[1] - dim[0]) / (pgd_steps * eps_multiplier))
                eps = []
                for d in hyperrectangle:
                    eps.append((d[1] - d[0]) / (pgd_steps * eps_multiplier))
                
                # Generate a random point from the hyperrectangle
                pgd_point = []
                for d in hyperrectangle:
                    pgd_point.append(np.random.uniform(d[0], d[1]))
                    
                # Convert the point and label to tensor
                pgd_point = tf.convert_to_tensor([pgd_point], dtype=tf.float32)
                pgd_label = tf.convert_to_tensor([hyperrectangles_labels[i]], dtype=tf.float32)

                # PGD loop
                for _ in range(pgd_steps):
                    with tf.GradientTape() as tape:
                        tape.watch(pgd_point)
                        prediction = model(pgd_point, training=False)
                        pgd_single_point_loss = pgd_attack_single_point_loss(pgd_label, prediction)
                    # Get the gradients of the loss w.r.t to the input point.
                    gradient = tape.gradient(pgd_single_point_loss, pgd_point)
                    # Get the sign of the gradients to create the perturbation
                    signed_grad = tf.sign(gradient)
                    pgd_point = pgd_point + signed_grad * eps
                    pgd_point = tf.clip_by_value(pgd_point, t_hyperrectangle[0], t_hyperrectangle[1])

                # Concatenate the pgd points
                if len(pgd_dataset) > 0:
                    pgd_dataset = np.concatenate((pgd_dataset, pgd_point), axis=0)
                    pgd_labels = np.concatenate((pgd_labels, pgd_label), axis=0)
                else:
                    pgd_dataset = pgd_point
                    pgd_labels = pgd_label

            pgd_dataset = np.asarray(pgd_dataset)
            pgd_labels = np.asarray(pgd_labels)

            # Convert the pgd generated inputs into tf datasets, shuffle and batch them
            pgd_dataset = tf.data.Dataset.from_tensor_slices((pgd_dataset, pgd_labels))
            pgd_dataset = pgd_dataset.shuffle(buffer_size=1024).batch(batch_size)

            # Iterate over the batches of the pgd dataset.
            for x_batch_train, y_batch_train in pgd_dataset: 
                # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    outputs = model(x_batch_train, training=True) # Outputs for this minibatch
                    pgd_loss_value = pgd_batch_loss(y_batch_train, outputs) # Calculate loss
                    pgd_loss_value = pgd_loss_value * beta # Multiply by 'beta'
                # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(pgd_loss_value, model.trainable_weights)
                # Run one step of gradient descent by updating the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
        ################################################################################################
        
        # Run a training loop at the end of each epoch.
        for x_batch_train, y_batch_train in train_dataset:
            train_outputs = model(x_batch_train, training=False)
            train_acc_metric.update_state(y_batch_train, train_outputs)
            train_loss_metric.update_state(y_batch_train, train_outputs)

        # Run a testing loop at the end of each epoch.
        for x_batch_test, y_batch_test in test_dataset:
            test_outputs = model(x_batch_test, training=False)
            test_acc_metric.update_state(y_batch_test, test_outputs)
            test_loss_metric.update_state(y_batch_test, test_outputs)

        # Run a pgd loop at the end of each epoch.
        for x_batch_test, y_batch_test in pgd_dataset:
            pgd_outputs = model(x_batch_test, training=False)
            pgd_acc_metric.update_state(y_batch_test, pgd_outputs)
            pgd_loss_metric.update_state(y_batch_test, pgd_outputs)

        train_acc = train_acc_metric.result()
        test_acc = test_acc_metric.result()
        pgd_acc = pgd_acc_metric.result()
        train_loss = train_loss_metric.result()
        test_loss = test_loss_metric.result()
        pgd_loss = pgd_loss_metric.result()

        train_acc_metric.reset_states()
        test_acc_metric.reset_states()
        pgd_acc_metric.reset_states()
        train_loss_metric.reset_states()
        test_loss_metric.reset_states()
        pgd_loss_metric.reset_states()

        train_acc = float(train_acc)
        test_acc = float(test_acc)
        pgd_acc = float(pgd_acc)

        train_loss = float(train_loss)
        test_loss = float(test_loss)
        pgd_loss = float(pgd_loss)

        print(f"Train acc: {train_acc:.4f}, Train loss: {train_loss:.4f} --- Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f} --- PGD acc: {pgd_acc:.4f}, PGD loss: {pgd_loss:.4f} --- Time: {(time.time() - start_time):.2f}s")
    return model