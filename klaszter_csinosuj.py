#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:18:59 2019

@author: szutor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#import pdal
import open3d as o3d
import os
import scipy.interpolate
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from  scipy.spatial import ConvexHull
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from scipy.interpolate import griddata
#import hdbscan
import datetime

kezd=datetime.datetime.now()
szoszatyar=False
plotkell=False
clusterviewkell=False

p_alaptalajszintelteres=0.5
szelvenymeret=30

talajszin=np.array([0.5,0.5,1])
epuletszin=[0.6,0.6,0.6]
erdoszin=[0.2,0.8,0.2]
#FFT viszgálat
def fftszamit(pcd,maxx,maxy):
    pcdt=np.asarray(pcd.points)
    interp= scipy.interpolate.LinearNDInterpolator(pcdt[:,(0,1)],pcdt[:,2], fill_value=0)
    xosztas=np.linspace(0,maxx)
    yosztas=np.linspace(0,maxy)
    X, Y = np.meshgrid(xosztas, yosztas)
    Z0 = interp(X, Y)
    #most fft-zünk
    fft_z = np.fft.fftshift(np.fft.fft2(Z0))
    fx = np.fft.fftshift(np.fft.fftfreq(xosztas.shape[0],xosztas[1]-xosztas[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(yosztas.shape[0],yosztas[1]-yosztas[0]))
    return (fx,fy,fft_z.real)

#osszehasonlit ket fft-t
def fftosztaly(fftminta,klaszpc,talajszint):
    
    vari=np.var(fftminta)
    t=1
    pontszam=klaszpc.shape[0]
       
    mut1=vari/pontszam
    ki=[0,0,0]
    minz=np.amin(klaszpc[:,2])
    maxz=np.amax(klaszpc[:,2])
    fftmax=np.amax(fftminta)
    fftmin=np.amin(fftminta)
    avgz=np.average(klaszpc[:,2]-talajszint)
    pcatlelter=abs(np.sum(klaszpc[:,2]-maxz))
    szoras=np.std(fftminta)
    mut2=(fftmax-fftmin)/pontszam
    mut3=szoras/pontszam
    mut4=np.average(fftminta)/pontszam
    zmeret=maxz-minz
    if szoszatyar:
        print('Variancia:',vari,' pontdarab:',pontszam,' Mut:',vari/pontszam)
        print('FFTmax:',fftmax,' FFTmin:',fftmin,'Mut2:',mut2)
        print('Szórás:',np.std(fftminta),' Mut3:',mut3)
        print('FFT avg',np.average(fftminta),' Mut4:',mut4)
        print('Avgz:',avgz)
    erdohatar=2.8

    if ((mut3<0.2 and mut4<0) and zmeret<6 and (minz-talajszint)>2 and avgz>3 and pontszam>100):
        ki=[1,0.4,0.4]
        #ki=epuletszin
    elif (fftmax>4000 and mut2<10 and mut3<0.20 and mut1<25):
        ki=[1,0.3,0.3]
        #ki=epuletszin
    elif (fftmax>3000 and mut2<11 and mut3<0.3):# erdoő
        #ki=[0.9,0.2,0.2]
        #ki=epuletszin
        ki=erdoszin
    elif  (avgz>3 and zmeret>2) : #erdő
        ki=[0.2,0.8,0.2]
        ki=erdoszin
    elif avgz>4:
        ki=[0.9,0.1,0.1] #megint épület
        #ki=epuletszin
    elif avgz>2 or mut1>40:
        ki=[0.2,0.8,0.2]
        ki=erdoszin
    elif pontszam<50:
        ki=[0.2,0.2,0.2]
    else:
        ki=talajszin #felesleges a további vizsgálat
    return ki

def fftkirajzol(ffttomb,pc,maxx,maxy):
    interp= scipy.interpolate.LinearNDInterpolator(pc[:,(0,1)],pc[:,2], fill_value=np.amin(pc[:,2]))
    xosztas=np.linspace(0,maxx)
    yosztas=np.linspace(0,maxy)
    X, Y = np.meshgrid(xosztas, yosztas)
    Z0 = interp(X, Y)
    #grid_z1 = griddata(pc[:,(0,1)], pc[:,2], (X, Y), method='linear')
    if szoszatyar and plotkell:
        print('pc min : ',np.amin(pc[:,2]),'pc max : ',np.amax(pc[:,2]))
        #fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(pc[:,0],pc[:,1],pc[:,2])
        plt.show()
        plt.pcolormesh(X,Y,Z0)
        plt.colorbar()         
        plt.show()
        plt.pcolormesh(ffttomb[0],ffttomb[1],ffttomb[2])
        plt.colorbar()  
        plt.show()
    return    


def kozelpontkeres(pont,pontokban): #bemenő paraméterek:[x,y,z],[[x,y,z],[]]
    tavok=np.linalg.norm(pont-pontokban,axis=1)
    kozeli=pontokban[np.argmin(tavok)]
    return kozeli  #a legközelbbi pont [x,y,z]

def kozelndarabkeres(pont,pontokban,mennyi): #bemenő paraméterek:[x,y,z],[[x,y,z],[]]
    keres=pontokban
    kozeliek=[]
    for i in range(0,mennyi):
        tavok=np.linalg.norm(pont-keres,axis=1)
        ind=np.argmin(tavok)
        kozeliek.append(keres[ind])
        keres=np.delete(keres,ind,axis=0)
    kozeliek=np.array(kozeliek)    
    return kozeliek 

###########################################[:,2]############
#beolvasas es klaszterezes
pcd = o3d.io.read_point_cloud("./szfvarmikro.ply",format='ply')


#kellenek a koordinatak
#a megjelenites maitt origoba transzformalunk
minbound=(pcd.get_min_bound())
maxbound=(pcd.get_max_bound())
trmap=[[1,0,0,-minbound[0]],[0,1,0,-minbound[1]],[0,0,1,0-minbound[2]],[0,0,0,1]]
pcd.transform(trmap)
minbound=(pcd.get_min_bound())
maxbound=(pcd.get_max_bound())
pckoor=np.asarray(pcd.points)
pcid=np.arange(0,len(pckoor))
#atlagos suruseg
pcsuruseg=pckoor.shape[0]/(maxbound[0]-minbound[0])*(maxbound[1]-minbound[1])

#a talaj megállapításához 10 méterenként megnézem a minimum pontot, ahhoz hasonlítom a klasztert, hogy talaj-e; a legközelebbi három közül a minimumot nézem, attól mennyir tér el
zminimumok=[]

for i in range(1,int(round(maxbound[0]/szelvenymeret))):
    for j in range(1,int(round(maxbound[1]/szelvenymeret))):
        szegmenspontok=[np.array([(i-1)*szelvenymeret,(j-1)*szelvenymeret,minbound[2]-1]),np.array([(i)*szelvenymeret,(j)*szelvenymeret,maxbound[2]+1])]
        szegmens=o3d.geometry.AxisAlignedBoundingBox(szegmenspontok[0],szegmenspontok[1])
        szelvenypc=pcd.crop(szegmens)
        szelveny=np.asarray(szelvenypc.points)
        #szelveny=pckoor[np.where(pckoor[:,0]>=(i-1)*10) and np.where(pckoor[:,0]<(i)*10) and np.where(pckoor[:,1]>=(j-1)*10) and np.where(pckoor[:,0]<(j)*10)]  
        #szelveny=pckoor[np.where(pckoor[:,0]>=(i-1)*10 and pckoor[:,0]<(i)*10 and pckoor[:,1]>=(j-1)*10 and pckoor[:,0]<(j)*10)] 
        if len(szelveny)>0:
            zminimumok.append(szelveny[np.argmin(szelveny[:,2])])
zminimumok=np.array(zminimumok) 
alaptalaj=np.zeros(len(pckoor),dtype=bool)
for i in range(0,len(pckoor)):
    talajszintek=kozelndarabkeres(pckoor[i],zminimumok,4)[:,2]
    for j in range(0,4):
        if (pckoor[i][2]-talajszintek[j])<p_alaptalajszintelteres:
            alaptalaj[i]=True
pckoortargy=pckoor[np.invert(alaptalaj)]
pckoortalaj=pckoor[alaptalaj]
pcdtalaj=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pckoortalaj))        
pcd2=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pckoortargy))
pcdtalaj.colors= o3d.utility.Vector3dVector(np.tile(talajszin-0.35,(len(pckoortalaj),1)))
#most klaszterezem eloszor
csopok=np.asarray(pcd2.cluster_dbscan(1.90,30,True))
#clusterer = hdbscan.HDBSCAN(min_cluster_size=40,alpha=1.12,).fit(np.asarray(pcd2.points))
#csopok=clusterer.labels_
                
#csoportokhoz veletlen szineket rendelek
csopdarab=(len(np.unique(csopok)))
szurke=np.linspace(0.1,0.9,csopdarab+1)
#colortable=np.array([szurke,szurke,szurke]).T
colortable=np.random.rand(csopdarab+1,3)
szinek=np.zeros((len(csopok),3),dtype='float64')
for i in range(0,len(csopok)):
    szinek[i]=colortable[csopok[i]]
    szinek[i,0]=colortable[csopok[i],0]
    szinek[i,1]=colortable[csopok[i],1]
    szinek[i,2]=colortable[csopok[i],2]
#kirajzoloma a DBSCAN klaszerezett csoportoka
pcd2.colors = o3d.utility.Vector3dVector(szinek)    
if clusterviewkell:
    o3d.visualization.draw_geometries([pcd2,pcdtalaj],window_name='DBSCAN klaszterezés után, véletlen színek')    
#a nagyobb klasztereket ujra osztom spectralclusteringgel
klaszterek=np.unique(csopok)
minz=np.amin(pckoortargy[:,2])
#a nagyobb klasztekeret tovabb klaszterezem      -- csak a befoglaló méretéhez képest kicsiket kell!!
csopok2=np.zeros((len(csopok),),dtype='int32')
for klasz in klaszterek:
    klaszkoor=pckoortargy[np.where(csopok==klasz)]
    if klaszkoor.shape[0]>10000 : # a kicsiket nem bontom tovább
        xmeret=np.amax(klaszkoor[:,0])-np.amin(klaszkoor[:,0])
        ymeret=np.amax(klaszkoor[:,1])-np.amin(klaszkoor[:,1])
        klaszsuruseg=klaszkoor.shape[0]/xmeret*ymeret
        if klaszsuruseg<pcsuruseg/10: # csak azokat, ahol a suruseg kicsi, tehát vannak távok
            pcidk=pcid[np.where(csopok==klasz)]
            brc = Birch(branching_factor=50, n_clusters=None, threshold=3.9, compute_labels=True)
            clustering=brc.fit(klaszkoor)        
            #print('2.clustering:',np.unique(clustering.labels_))
            csopok2[np.where(csopok==klasz)]=csopok2[np.where(csopok==klasz)]+clustering.labels_
print(len(csopok2),len(csopok))        
csopok=csopok*1000+csopok2    
print(len(csopok2),len(csopok))        
klaszterek=np.unique(csopok)
########################################################x
#ujra csinalom a szineket
csopdarab=(len(klaszterek))
szurke=np.linspace(0.1,0.9,csopdarab+1)
#colortable=np.array([szurke,szurke,szurke]).T
colortable=np.random.rand(csopdarab+1,3)
#szinek=np.zeros((len(csopok),3),dtype='float64')
for klasz in klaszterek:
    szinek[np.where(csopok==klasz)]=colortable[np.where(klaszterek==klasz)]
#kirajzoloma a klaszerezett csoportokat    
pcd2.colors = o3d.utility.Vector3dVector(szinek)    
if clusterviewkell:
    o3d.visualization.draw_geometries([pcd2,pcdtalaj],window_name='BIRCH klaszterezés után, véletlen színek')  
#####################################################

#külön rakom a talajokat
talaj=[]
klaszminek=[]

for klasz in klaszterek:
    klaszkoor=pckoortargy[np.where(csopok==klasz)]
    if np.amin(klaszkoor[:,2])==minz:
        talaj.append(klasz)
        szinek[np.where(csopok==klasz)]=talajszin-0.1
        klaszminek.append(minz)
    else:    
        #tavolasagok kiszamolasaa klaszter centroidtol
        # a minimumok kozul kell a koordinata
        #az X méternél közelebbik közül a legkisebb z-hez kell hasonlítani.
        try:
            klaszcentx=np.sum(klaszkoor[:,0],axis=0)/klaszkoor.shape[0]
            klaszcenty=np.sum(klaszkoor[:,1],axis=0)/klaszkoor.shape[0]
            a=np.linalg.norm(zminimumok[:,:2]-np.array([klaszcentx,klaszcenty]),axis=1)<szelvenymeret*3
            alsoz=np.amin(zminimumok[a,2])
        except:
            alsoz=minz
        #elrakom, mert kell később
        klaszminek.append(alsoz)
        #np.argmin(tavmat)
        klaszpc=o3d.geometry.PointCloud()
        klaszpc.points=o3d.utility.Vector3dVector(klaszkoor)
        minbound=(klaszpc.get_min_bound())
        maxbound=(klaszpc.get_max_bound())
        #trmap=[[1,0,0,-minbound[0]],[0,1,0,-minbound[1]],[0,0,1,0-minbound[2]],[0,0,0,1]]
        #klaszpc.transform(trmap)#innentől már 0,0,0 az origo
        maxx=maxbound[0]-minbound[0]
        maxy=maxbound[1]-minbound[1]    
        #ha lent van, talaj
        klaszatlag=np.average(klaszkoor[:,2])
        klaszminz=np.amin(klaszkoor[:,2])
        klaszmaxz=np.amax(klaszkoor[:,2])
        magassag=klaszmaxz-klaszminz
        if (abs(klaszatlag-minz)<0.4)  and magassag<2: #ennyi centi az aljatol
            talaj.append(klasz)
            szinek[np.where(csopok==klasz)]=talajszin
klaszminek=np.array(klaszminek)            
#print(talaj)
#####################################################
#klaszterenkent megvizsgalom az fft-t
if len(talaj)>0 or 1:
    for kli in range(0,len( klaszterek)):
        klasz=klaszterek[kli]
        if not klasz in talaj:
        
            klaszkoor=pckoortargy[np.where(csopok==klasz)]

            #if szoszatyar:
                #print('cluster id:',klasz,' pont darab:',len(klaszkoor))
            if len(klaszkoor)>30:
                klaszpc=o3d.geometry.PointCloud()
                klaszpc.points=o3d.utility.Vector3dVector(klaszkoor)
                minbound=(klaszpc.get_min_bound())
                maxbound=(klaszpc.get_max_bound())
                klaszmin=klaszminek[kli]
                trmap=[[1,0,0,-minbound[0]],[0,1,0,-minbound[1]],[0,0,1,0-minbound[2]],[0,0,0,1]]
                #Z-t rosszul transformál!! a talaj Z-jéhez kell hasonlítani!!!
                klaszpc.transform(trmap)#innentől már 0,0,0 az origo
                maxx=maxbound[0]-minbound[0]
                maxy=maxbound[1]-minbound[1]  
                klaszfft=fftszamit(klaszpc,maxx,maxy)
                fftkirajzol(klaszfft,np.asarray(klaszpc.points),maxx,maxy)
                szinek[np.where(csopok==klasz)]=fftosztaly(klaszfft[2],klaszkoor,klaszmin)
            else:
                szinek[np.where(csopok==klasz)]=[0.2,0.2,0.2]
########################################################x
#Végeredmény kirajzolás
pcd2.colors = o3d.utility.Vector3dVector(szinek)    
print(datetime.datetime.now()-kezd)
o3d.visualization.draw_geometries([pcd2,pcdtalaj],window_name='Kék:talaj,Zöld:erdő,Piros:nyeregtető,Szürke:lapostető')    
