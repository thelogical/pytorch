import pygame
import time
from qlearn import Q
import random
import pickle
import torch
import matplotlib.pyplot as plt


pygame.init()

height = 600
width = 800

gameDisplay = pygame.display.set_mode((800, 600))
pygame.display.set_caption('q learning')
clock = pygame.time.Clock()


game_grid = [[0,1,2,3,4,5,6,7,8],
             [9,10,11,12,13,14,15,16,17],
             [18,19,20,21,22,23,24,25,26]]

pits = [0,4,17,19,23]

goals = [6,14,20,25]

X = [[x] for x in range(0,27)]
Y = [[None],[10],[11],[12],[None],[14],[None],[6],[7],[10],[11],[20],[11],[14],[None],[14],[25],[None],[9],[None],[None],[20],[21],[None],[25],[None],[25]]



for c in X:
    if c[0] in goals:
        X.remove(c)

for c in X:
    if c[0] in pits:
        X.remove(c)

x = torch.tensor(X,dtype=torch.float)

y = torch.tensor([ [Y[c[0]][0] ] for c in X],dtype=torch.float)

N, D_in, H, D_out = len(X), 1, 50, 1

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.SmoothL1Loss()

learning_rate = 0.01
accuracy = 0
optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)

transition = [[False, False, True, True],
              [True, False, True, True],
              [True, False, True, True],
              [True, False, True, True],
              [True, False, True, True],
              [True, False, True, True],
              [True, False, True, True],
              [True, False, True, True],
              [True, False, False, True],
              [False, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
              [True, True, True, True],
              [True, True, False, True],
              [False, True, True, False],
              [True, True, True, False],
              [True, True, True, False],
              [True, True, True, False],
              [True, True, True, False],
              [True, True, True, False],
              [True, True, True, False],
              [True, True, True, False],
              [True, True, False, False],
                                        ]


class state:
    count = 0

    def __init__(self, x, y, v):
        self.start_x = x
        self.start_y = y
        self.value = v
        self.width = 70
        self.height = 100
        state.count += 1

    def show(self):
        print(str(self.start_x) + " " + str(self.start_y) + " " + str(self.value) + "\n")


class agent:
    def __init__(self):
        self.x = None
        self.y = None

    def moveTo(self, st):
        self.x = 50 + (st%9) * 70 + 35
        self.y = 200 + 80 + int(st/9)*100
        pygame.draw.circle(gameDisplay, (230, 0, 0), [self.x, self.y], 5)

    def erase(self):
        pygame.draw.circle(gameDisplay, (0, 0, 0), [self.x, self.y], 5)


def do_action(index,cur):
    if(index == 0):
        return cur - 1
    elif(index == 1):
        return cur - 9
    elif(index == 2):
        return cur + 1
    else:
        return cur + 9


def text_objects(text, font,color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()


def message_display(text, x, y,size,color):
    largeText = pygame.font.Font('freesansbold.ttf', size)
    TextSurf, TextRect = text_objects(text, largeText,color)
    TextRect.center = (x, y)
    gameDisplay.blit(TextSurf, TextRect)


def draw(State_list):
    for x in State_list:
        vl = x.value
        pygame.draw.rect(gameDisplay, (255, 255, 255),
                         [x.start_x, x.start_y, x.width, x.height], 5)
        message_display(str(vl), x.start_x + x.width / 2, x.start_y + x.height / 2,20,(0,255,0))



def initialize():
    # pygame.draw.rect(gameDisplay,(255,255,255),[100,250,500,100])
    State_list = []
    for row in game_grid:
        for val in row:
            State_list.append(state(50 + row.index(val) * 70, 200 + game_grid.index(row) * 100, val))
    draw(State_list)
    for x in pits:
        for st in State_list:
            if(st.value == x):
                message_display('PIT',st.start_x + 35,st.start_y + 70,10,(255, 182, 0))
    for x in goals:
        for st in State_list:
            if(st.value == x):
                message_display('GOAL',st.start_x + 35,st.start_y + 70,10,(255, 182, 0))
    pygame.display.update()

initialize()

gameExit = False

Ag = agent()

def get_index(old,new):
    if(new == old - 1):
        return 0
    elif(new == old - 9):
        return 1
    elif(new == old + 1):
        return 2
    else:
        return 3

x_val = []
y_val = []
acc = []
for ep in range(0,3000):
    cur = random.choice(random.choice(game_grid))
    end = False
    index = -1
    action = False
    while not end:
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        #print(ep, loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        action = False
        Ag.moveTo(cur)
        pygame.display.update()
        next_action = transition[cur]
        while action == False:
            index = random.randint(0,3)
            action = next_action[index]
        next_action = do_action(index,cur)
        cur = next_action
        if (cur in pits or cur in goals):
            end = True
            Ag.erase()
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        print("\n")
        Ag.erase()
        if(ep%100==0):
            y_test = model(x)
            delta = [[abs(float(y_test[c][0]) - float(y[c][0]))] for c in range(0, len(y))]
            accuracy = 0
            for i in range(0,len(y)):
                if(abs(float(y_test[i][0]) - float(y[i][0])) < 1):
                    accuracy+=1
            acc.append(float(accuracy)/float(len(y)))
            avg_delta = 0
            for c in delta:
                avg_delta = avg_delta + c[0]
            avg_delta /= len(y)
            x_val.append(ep)
            y_val.append(avg_delta)
        #clock.tick(1000)

plt.figure('1')
plt.plot(x_val,y_val)
plt.xlabel("Iterations")
plt.ylabel("Average Loss")
plt.figure('2')
plt.plot(x_val,acc)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.show()
pygame.quit()

#y_test = model(x)

#delta = [[abs(float(y_test[c][0])-float(y[c][0]))] for c in range(0,len(y))]
#avg_delta = 0
#for c in delta:
 #   avg_delta = avg_delta + c[0]
#avg_delta/=len(y)
#print(avg_delta)
#print(delta)

#with open('/home/cryptik/Desktop/q_values2.pkl', 'wb') as output:
    #lst = Q(q_values)
    #pickle.dump(lst, output, pickle.HIGHEST_PROTOCOL)