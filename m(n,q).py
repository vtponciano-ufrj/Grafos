# -*- coding: utf-8 -*-
"""M(n,q).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_mwWDubXA3P8_SLuAvF8IOIysxhzHw5M
"""

!pip install pyyed
import pyyed
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx
import csv

def ciclo_hamiltoniano_M(n,q):
 
  m=(n+q)//(2*(q))
  k=n//(q)
  h=q+(m-1)*(4*q-10)
            
  ################################# M(n,q) #################################################
  G =  pyyed.Graph()
  H =  pyyed.Graph()

  for i in range(1,n):
      G.add_edge('u'+str(i), 'v'+str(i),line_type="line",arrowhead="none", arrowfoot="none")
      G.add_edge('u'+str(i), 'u'+str(i+1),line_type="line",arrowhead="none", arrowfoot="none")
    
  G.add_edge('u'+str(1), 'u'+str(n),line_type="line",arrowhead="none", arrowfoot="none")   
  G.add_edge('u'+str(n), 'v'+str(n),line_type="line",arrowhead="none", arrowfoot="none")   
  

  def complete_graph(l):
     for i in range(1,k+1):
       for  j in range(l,q):
          G.add_edge('v'+str(i+(l-1)*k), 'v'+str(i+j*k),line_type="line",arrowhead="none", arrowfoot="none")   

  for i in range(1,q):
    complete_graph(i)     
  G.write_graph('M('+str(n)+','+str(q)+')_'+'Grafo.graphml', pretty_print=True)
 
for i in range(3,11):
  ciclo_hamiltoniano_M(1000,i)