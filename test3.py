#--- параметры ---
n=3; size=50; p=12
#-----------------
from time import *
from random import *
# импорт графической библиотеки Питона
from tkinter import *
from tkinter.messagebox import *
# создание окна приложения с "холстом" для рисования
root=Tk()
root.title('Найди пару')
canvas=Canvas(root, width=2*p+2*n*size,
 height=2*p+n*size,bg='lightgray')
canvas.pack()


class Cell:
    def __init__(self,i,j,k):
        self.i=i
        self.j=j
        self.k=k
        x=p+self.j*size
        y=p+self.i*size
        self.fon=canvas.create_rectangle(
            (x+3,y+3),(x+size-3,y+size-3),
            outline='', fill='#cccccc')
        self.num = canvas.create_text(
            (x+size//2,y+size//2),
            font=('Courier New',22,'bold'))


    def change(self,s):
        self.s=s
        if self.s==0:
            canvas.itemconfig(self.fon, 
            outline='black', fill='#88cc88')
            canvas.itemconfig(self.num, text='')
        if self.s==1:
            canvas.itemconfig(self.fon, 
            outline='black', fill='yellow')
            canvas.itemconfig(self.num, text=str(self.k))
        if self.s==-1:
            canvas.itemconfig(self.fon,
            outline='', fill='#cccccc')
            canvas.itemconfig(self.num, text='')


def newgame():
    global state, count
    count=2*n*n
    state=0
    L=[k+1 for k in range(n*n)]
    # L = sample(range(100), n*n)
    L.extend(L); shuffle(L)
    k=0
    for i in range(n):
        for j in range(2*n):
            mcell[i][j].k=L[k]
            mcell[i][j].change(1)
            canvas.update()
            sleep(0.2)
            mcell[i][j].change(0)
            k+=1


def reaction(event):
    global state, ns, np, count
    i=(event.y-p)//size
    j=(event.x-p)//size
    if 0<=i<n and 0<=j<2*n:
        if state==0 and mcell[i][j].s==0:
    # обработка первого щелчка
            mcell[i][j].change(1)
            ns=i; np=j
            state=1
        elif state==1 and mcell[i][j].s==0:
            # обработка второго щелчка
            mcell[i][j].change(1)
            canvas.update()
            #root.after(1000,None)
            sleep(1)
            if mcell[i][j].k==mcell[ns][np].k:
                mcell[i][j].change(-1)
                mcell[ns][np].change(-1)
                state=0
                count-=2
            if count==0:
                res=askyesno('Найди пару','Играть ещё?')
                if res: newgame()
                else: root.destroy()
        else:
            mcell[i][j].change(0)
            mcell[ns][np].change(0)
            state=0


#------- main ------
mcell=[]

for i in range(n):
    w=[]
    for j in range(2*n):
        w.append(Cell(i,j,-1))
    mcell.append(w)
start_time = time()

newgame()
canvas.bind('<Button-1>', reaction)
root.mainloop()
print("--- %s seconds ---" % (time() - start_time))