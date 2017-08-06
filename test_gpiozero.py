from gpiozero import Button

next_overlay_btn = Button(15)
take_pic_btn = Button(18)

def next_overlay():
    print("Next overlay")

def take_picture():
    print("Take a picture")

while True:
 next_overlay_btn.when_pressed = next_overlay
 take_pic_btn.when_pressed = take_picture