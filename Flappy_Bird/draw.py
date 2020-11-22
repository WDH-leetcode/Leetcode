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
white = [255, 255, 255 ]
clock = pygame.time.Clock()
run = True
mouse_pressed = False
while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        x, y = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1, 0, 0):
            pygame.draw.rect(screen,(255, 255, 255), (x, y, 10, 10))
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
        pygame.display.update()
        clock.tick(1000)
fname = 'number.jpg'
pygame.image.save(screen, fname)
print('the number has been saved as {}'.format(fname))
img = Image.open('number.jpg').convert('L')
pyplot.figure()
pyplot.plot(img)
pyplot.show()
'''y = np.asarray(img.getdata(), dtype=np.float64).reshape((img.size[1], img.size[0]))
y = np.asarray(y, dytpe=np.uint8)
digi_img = Image.fromarray(y, mode='L')
digi_img.save('digitalized image.jpg')'''
pygame.quit()
