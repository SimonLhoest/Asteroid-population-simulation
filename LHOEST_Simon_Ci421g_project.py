# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:27:19 2023

@author: LHOEST Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def Circular(u,t): #Trajectory of asteroid with Jupiter on circular orbit
    ju=np.array([5.2*np.cos(2*np.pi*t/T), 5.2*np.sin(2*np.pi*t/T), 0])
    return np.array([u[3], u[4], u[5], 
                     -G*mo/((u[0]**2+u[1]**2+u[2]**2)**(3/2))*u[0]-G*mj*((u[0]-ju[0])/(((u[0]-ju[0])**2+(u[1]-ju[1])**2+(u[2]-ju[2])**2)**(3/2))+ju[0]/(ju[0]**2+ju[1]**2+ju[2]**2)**(3/2)), 
                     -G*mo/((u[0]**2+u[1]**2+u[2]**2)**(3/2))*u[1]-G*mj*((u[1]-ju[1])/(((u[0]-ju[0])**2+(u[1]-ju[1])**2+(u[2]-ju[2])**2)**(3/2))+ju[1]/(ju[0]**2+ju[1]**2+ju[2]**2)**(3/2)),  
                     -G*mo/((u[0]**2+u[1]**2+u[2]**2)**(3/2))*u[2]-G*mj*((u[2]-ju[2])/(((u[0]-ju[0])**2+(u[1]-ju[1])**2+(u[2]-ju[2])**2)**(3/2))+ju[2]/(ju[0]**2+ju[1]**2+ju[2]**2)**(3/2))])


def Realistic(u,t): #Function with calculus of jupiter realistic trajectory
    M=Mo+n*(t)
    E=Mo
    while abs(E-ecc*np.sin(E)-M)>10**(-6):
        E=E-(E-ecc*np.sin(E)-M)/(1-ecc*np.cos(E))
    
    X=aj*(np.cos(E)-ecc)
    Y=aj*(1-ecc**2)**(1/2)*np.sin(E)
    
    jusec=r3Omega@r1inc@r3omega@np.array([[X],[Y],[0]])
    ju[0]=jusec[0]
    ju[1]=jusec[1]
    ju[2]=jusec[2]
    return np.array([u[3], u[4], u[5], 
                     -G*mo/((u[0]**2+u[1]**2+u[2]**2)**(3/2))*u[0]-G*mj*((u[0]-ju[0])/(((u[0]-ju[0])**2+(u[1]-ju[1])**2+(u[2]-ju[2])**2)**(3/2))+ju[0]/(ju[0]**2+ju[1]**2+ju[2]**2)**(3/2)), 
                     -G*mo/((u[0]**2+u[1]**2+u[2]**2)**(3/2))*u[1]-G*mj*((u[1]-ju[1])/(((u[0]-ju[0])**2+(u[1]-ju[1])**2+(u[2]-ju[2])**2)**(3/2))+ju[1]/(ju[0]**2+ju[1]**2+ju[2]**2)**(3/2)),  
                     -G*mo/((u[0]**2+u[1]**2+u[2]**2)**(3/2))*u[2]-G*mj*((u[2]-ju[2])/(((u[0]-ju[0])**2+(u[1]-ju[1])**2+(u[2]-ju[2])**2)**(3/2))+ju[2]/(ju[0]**2+ju[1]**2+ju[2]**2)**(3/2))])


def R1(O): #Construction of R1 matrix
    return np.array([[1, 0, 0],
                     [0, np.cos(O), np.sin(O)],
                     [0, -np.sin(O), np.cos(O)]])
def R3(O): #Construction of R3 matrix
    return np.array([[np.cos(O), np.sin(O), 0],
                     [-np.sin(O), np.cos(O), 0],
                     [0, 0, 1]])


#%% BACK AND FORTH TO DISPLAY ERROR 

h=0.1 #time step, in day
G=(2.95824*10**-4) #Gravitational constant
A=3.5 #Initial semi-major axis
mo=1 #Solar mass
t=np.arange(0,36500+h,h) #time 100 years
u=np.zeros((t.shape[0],6)) #position and velocity of asteroid
u[0]=np.array([A, 0, 0, 0, (G/A)**(1/2),0]) #Initiate u
ju=np.array([0, 0, 0]) #Jupiter position
mj=1/1047.348625 #Jupiter mass
T=4331.797849 #Period Jupiter

f=Circular

#ONE WAY
for i in range(t.shape[0]-1):
   k1=f(u[i],t[i])
   k2=f(u[i]+h/2*k1,t[i]+h/2)
   k3=f(u[i]+h/2*k2,t[i]+h/2)
   k4=f(u[i]+h*k3,t[i]+h)
   u[i+1]=u[i]+h/6*(k1+2*k2+2*k3+k4)
   
#STORE IN AND DO AGAIN
u2=u.copy()
h=-h
#GO BACK
for i in range(t.shape[0]-1,0,-1):
    k1=f(u[i],t[i])
    k2=f(u[i]+h/2*k1,t[i]+h/2)
    k3=f(u[i]+h/2*k2,t[i]+h/2)
    k4=f(u[i]+h*k3,t[i]+h)
    u[i-1]=u[i]+h/6*(k1+2*k2+2*k3+k4)

#Show error
print("Error induced by Runge-Kutta method : ",np.linalg.norm(u2[0,:3]-u[0,:3]))
    
#%% Asteroid perturbated by Jupiter on circular orbit 

A=3.5#Semi-major axis
h=1 #time step
t=np.arange(0,3650*2+h,h) #time 20years
u=np.zeros((t.shape[0],6)) #position and velocity of asteroid
u[0]=np.array([A, 0, 0, 0, (G/A)**(1/2),0]) #Initiate u
#Store semi-major axis, eccentricity and inclination
a=np.zeros((t.shape[0]))
e=np.zeros((t.shape[0]))
j=np.zeros((t.shape[0]))
a[0]=A
e[0]=0
j[0]=0

f=Circular

#Runge-Kutta to calculate trajectory
for i in range(t.shape[0]-1):
   k1=f(u[i],t[i])
   k2=f(u[i]+h/2*k1,t[i]+h/2)
   k3=f(u[i]+h/2*k2,t[i]+h/2)
   k4=f(u[i]+h*k3,t[i]+h)
   u[i+1]=u[i]+h/6*(k1+2*k2+2*k3+k4)
   
   r=u[i,:3]
   rp=u[i,3:]
   nr=np.linalg.norm(r)
   nrp=np.linalg.norm(rp)
   a[i+1]=(2/nr-(nrp**2)/(G*mo))**(-1)
   et=np.cross(rp,np.cross(r,rp))/(G*mo)-r/nr
   e[i+1]=np.linalg.norm(et)
   k=np.cross(r,rp)/(nr*nrp)
   j[i+1]=np.arccos(k[2])*(180/np.pi)
   
#Display plots
plt.figure('Semi-major axis with Jupiter on a circular orbit')
plt.plot(t,a)
plt.figure('Eccentricity with Jupiter on a circular orbit')
plt.plot(t,e)
plt.figure('Inclination with Jupiter on a circular orbit')
plt.plot(t,j)
plt.show()

#Store circular to compare with realistic later
ucircular=u.copy()
acircular=a.copy()
ecircular=e.copy()
icircular=j.copy()

#%% Asteroid pertubated by Jupiter on realistic orbit

#Orbital elements of Jupiter
aj=5.202575 #Semi-major axis
ecc=0.048908 #Eccentricity
inc=1.3038*np.pi/180 #Inclination
Omega=100.5145*np.pi/180 #Longitude of ascending node
omega=274.8752*np.pi/180 # argument of periapsis
Mo=80.0392*np.pi/180 #Mean anomaly
mj=1/1047.348625 #Mass Jupiter
T=4331.797849 #Period Jupiter
ju=np.array([0, 0, 0]) #Jupiter traj
K=G**(1/2)
n=K/((aj**3)**(1/2)) #Mean motion
r3Omega=R3(-Omega)
r1inc=R1(-inc)
r3omega=R3(-omega)

u=np.zeros((t.shape[0],6))
u[0]=np.array([A, 0, 0, 0, (G/A)**(1/2),0])
a=np.zeros((t.shape[0]))
e=np.zeros((t.shape[0]))
j=np.zeros((t.shape[0]))
a[0]=A
e[0]=0
j[0]=0



f=Realistic

for i in range(t.shape[0]-1):
   k1=f(u[i],t[i])
   k2=f(u[i]+h/2*k1,t[i]+h/2)
   k3=f(u[i]+h/2*k2,t[i]+h/2)
   k4=f(u[i]+h*k3,t[i]+h)
   u[i+1]=u[i]+h/6*(k1+2*k2+2*k3+k4)
   
   r=u[i,:3]
   rp=u[i,3:]
   nr=np.linalg.norm(r)
   nrp=np.linalg.norm(rp)
   a[i+1]=(2/nr-(nrp**2)/(G*mo))**(-1)
   et=np.cross(rp,np.cross(r,rp))/(G*mo)-r/nr
   e[i+1]=np.linalg.norm(et)
   k=np.cross(r,rp)/(nr*nrp)
   j[i+1]=np.arccos(k[2])*(180/np.pi)
   k=np.cross(r,rp)/(nr*nrp)
   j[i+1]=np.arccos(k[2])*(180/np.pi)
   
#Display plot
plt.figure("Semi-major axis with Jupiter on a realistic vs circular orbit")
plt.title("Semi-major axis with Jupiter on a realistic vs circular orbit")
plt.plot(t,a,label="Realistic")
plt.plot(t,acircular,label="Circular")
plt.legend()
plt.figure("Eccentricity with Jupiter on a realistic vs circular orbit")
plt.title("Eccentricity with Jupiter on a realistic vs circular orbit")
plt.plot(t,e,label="Realistic")
plt.plot(t,ecircular,label="Circular")
plt.legend()
plt.figure("Inclination with Jupiter on a realistic vs circular orbit")
plt.title("Inclination with Jupiter on a realistic vs circular orbit")
plt.plot(t,j,label="Realistic")
plt.plot(t,icircular,label="Circular")
plt.legend()
plt.show()

#%% Integration of 2007VW266 perturbated by realistic Jupiter

h=2 #Put 2 so we have faster processing time, 1000year is a lot
G=(2.95824*10**-4)
k=G**(1/2)
mo=1
epoch=2456600.5
t=np.arange(0+epoch,epoch+365000+h,h) #1000 years
u=np.zeros((t.shape[0],6))

#Orbital elements of asteroid
A=5.454 #Semi-major axis 
ecca=0.3896 #Eccentricity
inca=108.358 #Inclination
Omegaa=276.509*np.pi/180#Longitude of ascending node
omegaa=226.107*np.pi/180#Argument of periapsis
Moa=146.88*np.pi/180#Mean anomaly
TT=(4*np.pi**2*A**3/(G*mo))**(1/2) #Period of asteroid

#Initial position of asteroids
na=k/((A**3)**(1/2))
Ma=Moa+na*(epoch-epoch)#Mean motion
Ea=Moa
#Calculus of eccentric anomaly
while abs(Ea-ecca*np.sin(Ea)-Ma)>10**(-6):
    Ea=Ea-(Ea-ecca*np.sin(Ea)-Ma)/(1-ecca*np.cos(Ea))

#Find x y z and vx vy vz
ra=A*(1-ecca*np.cos(Ea))
Xa=A*(np.cos(Ea)-ecca)
Ya=A*(1-ecca**2)**(1/2)*np.sin(Ea)
Xpa=-na*A**2/ra*np.sin(Ea)
Ypa=na*A**2/ra*(1-ecca**2)**(1/2)*np.cos(Ea)
ar3Omega=R3(-Omegaa)
ar1inc=R1(-inca)
ar3omega=R3(-omegaa)
u[0,:3]=(ar3Omega@ar1inc@ar3omega@np.array([[Xa],[Ya],[0]])).reshape((3,))
u[0,3:]=(ar3Omega@ar1inc@ar3omega@np.array([[Xpa],[Ypa],[0]])).reshape((3,))

a=np.zeros((t.shape[0]))
e=np.zeros((t.shape[0]))
j=np.zeros((t.shape[0]))
a[0]=A
e[0]=0.3896
j[0]=108.358

f=Realistic

for i in range(t.shape[0]-1):
   k1=f(u[i],t[i])
   k2=f(u[i]+h/2*k1,t[i]+h/2)
   k3=f(u[i]+h/2*k2,t[i]+h/2)
   k4=f(u[i]+h*k3,t[i]+h)
   u[i+1]=u[i]+h/6*(k1+2*k2+2*k3+k4)
   
   r=u[i,:3]
   rp=u[i,3:]
   nr=np.linalg.norm(r)
   nrp=np.linalg.norm(rp)
   a[i+1]=(2/nr-(nrp**2)/(G*mo))**(-1)
   et=np.cross(rp,np.cross(r,rp))/(G*mo)-r/nr
   e[i+1]=np.linalg.norm(et)
   k=np.cross(r,rp)/(nr*nrp)
   j[i+1]=np.arccos(k[2])*(180/np.pi)
  
#Calculus of Jupiter trajectory
def Jup(t):
    M=Mo+n*(t-epoch)
    E=Mo
    while abs(E-ecc*np.sin(E)-M)>10**(-6):
        E=E-(E-ecc*np.sin(E)-M)/(1-ecc*np.cos(E))
    
    X=aj*(np.cos(E)-ecc)
    Y=aj*(1-ecc**2)**(1/2)*np.sin(E)
    
    return (r3Omega@r1inc@r3omega@np.array([[X],[Y],[0]])).reshape((3,))

#Store jupiter trajectory
jupiter=np.zeros((t.shape[0],3))
for i in range(t.shape[0]):
    jupiter[i]=Jup(t[i])

#Display plot
ax = plt.figure("3d pojection of the trajectories").add_subplot(projection='3d')
ax.plot(u[:,0],u[:,1],u[:,2],label="2007VW266")
ax.plot(jupiter[:,0],jupiter[:,1],jupiter[:,2],label="Jupiter")
plt.legend()
t=(t-epoch)/365.25 #To display in years instead of days 
plt.figure("Semi-major axis of 2007VW266")
plt.title("Semi-major axis of 2007VW266")
plt.plot(t,a)
plt.xlabel("t (year)")
plt.figure("Eccentricity of 2007VW266")
plt.title("Eccentricity of 2007VW266")
plt.plot(t,e)
plt.xlabel("t (year)")
plt.figure("Inclination of 2007VW266 !First value problem!")
plt.title("Inclination of 2007VW266")
plt.plot(t[1:],j[1:])
plt.xlabel("t (year)")
plt.show()

#Save for comparaison with clones
aog=a.copy()
eog=e.copy()
jog=j.copy()

#%% CLones of 2007VW266

#Incertainty of measurments
dA=0.0156 #Delta of semi-major axis
decca=0.00170#Delta of eccentricity
dinca=0.0261 #delta of inclination
dOmegaa=0.001144*np.pi/180 #Delta of longitude of ascending node
domegaa=0.0501*np.pi/180 #Delta of argument of periapsis
dMoa=0.604*np.pi/180 #Delta of Mean anomly
t=np.arange(0+epoch,epoch+365000+h,h)#1000years

#Same process as before for each clone.
f=Realistic
Nclones=3
uclones=np.zeros((t.shape[0],6,Nclones))
rnd.seed(42)
for clone in range(Nclones):
    #Take value included in the approximation fork
    A=rnd.uniform(A-dA,A+dA)
    ecca=rnd.uniform(ecca-decca,ecca+decca)
    inca=rnd.uniform(inca-dinca,inca+dinca)
    Omegaa=rnd.uniform(Omegaa-dOmegaa,Omegaa+dOmegaa)
    omegaa=rnd.uniform(omegaa-domegaa,omegaa+domegaa)
    Moa=rnd.uniform(Moa-dMoa,Moa+dMoa)
    
    na=K/((A**3)**(1/2))
    Ma=Moa+na*(epoch-epoch)
    Ea=Moa
    while abs(Ea-ecca*np.sin(Ea)-Ma)>10**(-6):
        Ea=Ea-(Ea-ecca*np.sin(Ea)-Ma)/(1-ecca*np.cos(Ea))
    ra=A*(1-ecca*np.cos(Ea))
    Xa=A*(np.cos(Ea)-ecca)
    Ya=A*(1-ecca**2)**(1/2)*np.sin(Ea)
    Xpa=-na*A**2/ra*np.sin(Ea)
    Ypa=na*A**2/ra*(1-ecca**2)**(1/2)*np.cos(Ea)
    ar3Omega=R3(-Omegaa)
    ar1inc=R1(-inca)
    ar3omega=R3(-omegaa)
    u[0,:3]=(ar3Omega@ar1inc@ar3omega@np.array([[Xa],[Ya],[0]])).reshape((3,))
    u[0,3:]=(ar3Omega@ar1inc@ar3omega@np.array([[Xpa],[Ypa],[0]])).reshape((3,))

    #Initiate arrays for a e inc
    a=np.zeros((t.shape[0]))
    e=np.zeros((t.shape[0]))
    j=np.zeros((t.shape[0]))
    a[0]=A
    e[0]=ecca
    j[0]=inca
    for i in range(t.shape[0]-1):
       k1=f(u[i],t[i])
       k2=f(u[i]+h/2*k1,t[i]+h/2)
       k3=f(u[i]+h/2*k2,t[i]+h/2)
       k4=f(u[i]+h*k3,t[i]+h)
       u[i+1]=u[i]+h/6*(k1+2*k2+2*k3+k4)
       
       r=u[i,:3]
       rp=u[i,3:]
       nr=np.linalg.norm(r)
       nrp=np.linalg.norm(rp)
       a[i+1]=(2/nr-(nrp**2)/(G*mo))**(-1)
       et=np.cross(rp,np.cross(r,rp))/(G*mo)-r/nr
       e[i+1]=np.linalg.norm(et)
       k=np.cross(r,rp)/(nr*nrp)
       j[i+1]=np.arccos(k[2])*(180/np.pi)
    uclones[:,:3,clone]=u[:,:3]
    uclones[:,3,clone]=a
    uclones[:,4,clone]=e
    uclones[:,5,clone]=i
    
t=(t-epoch)/365.25 #To display in years instead of days 
#Display parameters with eah clones
option=3
ax = plt.figure("Semi-major axis of clones and original 2007VW266").add_subplot()
for i in range (Nclones):
    ax.plot(t,uclones[:,option,i],label="clone "+str(i+1))
ax.plot(t,aog,label="og")
plt.ylabel("Semi-major axis (au)")
plt.xlabel("t (year)")
plt.grid()
ax.legend()
plt.show()

option=4
ax = plt.figure("Eccentricity of clones and original 2007VW266").add_subplot()
for i in range (Nclones):
    ax.plot(t,uclones[:,option,i],label="clone"+str(i+1))
ax.plot(t,eog,label="og")
plt.ylabel("Eccentricity")
plt.xlabel("t (year)")
plt.grid()
ax.legend()
plt.show()

#Inclination doesn't work.
# option=5
# ax = plt.figure("Inclination of clones and original 2007VW266").add_subplot()
# for i in range (Nclones):
#     ax.plot(t[1:],uclones[1:,option,i],label="clone"+str(i+1))
# ax.plot(t[1:],jog[1:],label="og")
# plt.ylabel("Inclination (deg)")
# plt.xlabel("t (year)")
# plt.grid()
# ax.legend()
# plt.show()


    









    