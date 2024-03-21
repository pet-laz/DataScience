from tkinter import *

def move_chicken():
    c.move(chicken_body, 5, 0)
    c.move(chicken_head, 5, 0)
    c.move(chicken_leg1, 5, 0)
    c.move(chicken_leg2, 5, 0)
    if c.coords(chicken_body)[0] < 500:
        root.after(50, move_chicken)


root = Tk()
c = Canvas(width=500, height=500, bg='DeepSkyBlue')
c.pack()
# c.create_line(10, 10, 190, 150)
# c.create_rectangle(10,10,190,150,width=5,fill='blue')
# c.create_line(10, 10, 190, 150,width=3,fill='orange')
# c.create_polygon(5,195,100,5,195,195,fill='black')
# c.create_oval(10, 10, 190, 150,width=3,fill='orange')

mdash = c.create_oval(0, 0, 30, 30, width=1,
                      fill='yellow')
mdash1 = c.create_rectangle(20, 90, 180, 150,
                   fill='brown')
c.create_rectangle(150, 45, 170, 90, width=2,
                   fill='brown')
c.create_polygon(100, 10, 20, 90, 180, 90,
                 fill='red', outline='black')
c.create_rectangle(0, 150, 500, 500,
                   fill='chartreuse2')
c.create_rectangle(70, 100, 130, 140,
                   fill='royal blue')
c.create_line(100, 140, 100, 100,
              fill='brown', width=3)
c.create_line(60, 120, 180, 120,
              fill='brown', width=3)
#ДЕРЕВО
c.create_rectangle(170, 125, 180, 180,
                   fill='brown')
c.create_oval(150, 100, 200, 150, width=1,
              fill='green')
#ВТОРОЕ ДЕРЕВО
c.create_rectangle(263, 135, 271, 170,
                   fill='brown')
c.create_oval(245, 90, 290, 140, width=1,
              fill='green')
#ТРЕТЬЕ ДЕРЕВО
c.create_rectangle(210, 270, 230, 170,
                   fill='brown')
c.create_oval(180, 120, 260, 200, width=1,
              fill='green')

c.create_rectangle(83, 50, 117, 70,
                   fill='royalblue')
c.create_line(83, 60, 117, 60,
              fill='brown', width=3)
c.create_line(100, 50, 100, 70,
              fill='brown', width=3)
c.create_rectangle(8, 155, 53, 190,
                   fill='brown')
c.create_oval(20, 165, 40, 185, width=1,
              fill='grey')
c.create_polygon(30, 120, 8, 155, 53, 155,
                 fill='red', outline='black')
#ВТОРОЙ ДОМ
c.create_rectangle(250, 300, 450, 400,
                   fill='brown')
c.create_polygon(250,300,350,200,450,300,
                 fill='red', outline='black')
c.create_rectangle(270,320, 310,370,
                   fill='royalblue')
c.create_rectangle(330,320, 370,370,
                   fill='royalblue')
c.create_rectangle(390,320,430,370,
                   fill='royalblue')

chicken_body = c.create_oval(100, 400, 150, 450, fill='yellow')
chicken_head = c.create_oval(150, 410, 170, 430, fill='orange')
chicken_leg1 = c.create_line(105, 450, 105, 470, width=3)
chicken_leg2 = c.create_line(145, 450, 145, 470, width=3)

def mmove():
    c.move(mdash, 10, 10)
    c.after(500, mmove)


move_chicken()

root.mainloop()