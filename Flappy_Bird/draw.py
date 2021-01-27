import pygame
from PIL import Image
from matplotlib import pyplot
import numpy as np
import sys
import tensorflow as tf
# create a drawing screen
pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption('Please draw a number.')
white = [255, 255, 255]
clock = pygame.time.Clock()
run = True
mouse_pressed = False
while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        x, y = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1, 0, 0):
            pygame.draw.rect(screen, (255, 255, 255), (x, y, 10, 10))
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
        pygame.display.update()
        clock.tick(1000)
fname = 'number.jpg'
pygame.image.save(screen, fname)
print('the number has been saved as {}'.format(fname))
img = Image.open('number.jpg').convert('L')
img = img.resize((28, 28))
img.save('greyscale.png')
img2arr = np.asarray(img) / 255
print(img2arr)
img2arr = img2arr.reshape(1, 28, 28)

### training data using built in dataset
mnist = tf.keras.datasets.mnist
(training_data, training_labels), (test_data, test_labels) = mnist.load_data()
training_data, test_data = training_data / 255, test_data / 255
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=5)
prediction = model.predict_classes(img2arr)
print(prediction[0])
pygame.quit()
