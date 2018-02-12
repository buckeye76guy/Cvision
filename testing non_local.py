# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:59:45 2018

@author: Josiah Hounyo
"""

def plus():
    index = 0
    
    def add():
        nonlocal index
        index += 1
        
        print('inner:', index)
    for i in range(12):
        add()
    print('outter:', index)