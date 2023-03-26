#!/usr/bin/env python
# coding: utf-8

# In[9]:


# importer les fichiers


import streamlit as st
st.set_page_config(layout="wide", page_icon = ':basketball:')
import joblib
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from matplotlib import cm
sns.set_theme()
from matplotlib.patches import Circle, Rectangle, Arc
pd.set_option('display.max_columns', None)

import ipywidgets as widgets
from matplotlib.colors import ListedColormap
from  matplotlib.colors import LinearSegmentedColormap



                                                    # récupérer les données
@st.cache_data
def open_df(dataframe):
    return pd.read_csv(dataframe).loc[0:100]

#dataset1
os.chdir("C:/Users/antho/Projet_NBA/Dataset1_Tirs de NBA entre 1997 et 2019")
df1 = open_df('NBA Shot Locations 1997 - 2020.csv')

#dataset2
os.chdir("C:/Users/antho/Projet_NBA/Dataset2_Actions de chaque match entre 2000 et 2020")
df00_01 = open_df('2000-01_pbp.csv')

#dataset3
os.chdir("C:/Users/antho/Projet_NBA/Dataset3_Bilans d'équipe entre 2014 et 2018")
df4 = open_df('games.csv')
df5 = open_df('games_details.csv')
df6 = open_df('teams.csv')
df7 = open_df('players.csv')
df8 = open_df('ranking.csv')

#dataset4
os.chdir("C:/Users/antho/Projet_NBA/Dataset4_Joueurs de NBA depuis 1950")
df9 = open_df('player_data.xls')
df10 = open_df('Players.xls')
df11 = open_df('Seasons_Stats.csv')

# datasets finaux
os.chdir("C:/Users/antho/Projet_NBA")
df_model = pd.read_pickle('Nettoyage_données_NBA.pkl')
df_viz = pd.read_pickle('Nettoyage_données_NBA_DataViz.pkl')
df_viz['position'].replace({1,2,3,4,5},{'meneur de jeu','arrière','ailier shooter','ailier fort','pivot'}, inplace = True)
df_viz['mène'].replace({-1,0,1},{'mené','égalité','mène'}, inplace = True)

# In[12]:
st.title(':dark[MSPy : ANALYSE DES TIRS DE JOUEURS NBA]')


st.write('_Projet présenté par Etienne Allemand; Adrian Chmielewski, Sora Derraz et Anthony Ferre_')
st.write('_Tuteur de projet : Dimitri Condoris_')

choix_partie = st.sidebar.radio("Sommaire",["I - Introduction","II - Exploration des données","III - Data visualization","IV - Analyse des tirs de 20 des meilleurs joueurs de NBA du 21ème siècle","V - Modèle de prédiction des tirs de joueurs de NBA", "VI - Test du modèle sur de nouvelles données", "VII - Conclusion et Perspectives"])

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
# If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                                  fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                                  fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,bottom_free_throw, restricted, corner_three_a,corner_three_b, three_arc, center_outer_arc,center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    # Set axis limits
    ax.set_xlim(-250, 250)
    ax.set_ylim(-50, 470)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    # General plot parameters
    mpl.rcParams['font.family'] = 'Avenir'
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.linewidth'] = 2

    return ax




                                        #Partie 1 : Introduction
    
if choix_partie == 'I - Introduction':
    st.subheader('I - Introduction')
    img_1 = Image.open("evolution_tir.png")
    st.image(img_1, caption = '(gauche) Évolution du taux de shots à 3 points (en bleu) et 2 points (en rouge) par match entre 1997 et 2020 (droite) Taux de shoots par match en fonction des différentes zones du terrain')
    st.subheader('Deux principaux objectifs :')
    st.markdown(''':dart: Comparer les tirs de vingt des meilleurs joueurs NBA du 21ème siècle cités ci-dessous : 
- Tim Duncan ;
- Kobe Bryant ;
- Allen Iverson ;
- Steve Nash ;
- Ray Allen ;
- etc...''')
    st.markdown(''':dart: Construire un modèle permettant de prédire si le tir de l'un des 10 joueurs encore actifs est réussi ou non à partir de paramètres (distance du tir, angle, type de tir, temps restant, niveau d'adversité, etc)''')
                                        # Partie 2 : Analyse des datasets originaux disponibles

    
if choix_partie == "II - Exploration des données":

    st.subheader("II - Exploration des données")
    pd.set_option('display.max_columns', None)
    
    choix_dataset = st.selectbox('Sélectionner le dataset d intérêt',['dataset_1','dataset_2','dataset_3','dataset_4'])
                                                    
        
                                                        # dataset 1
            
    if choix_dataset == 'dataset_1':
        
        choix_df = st.radio('Sélectionner le df d intérêt',['df1'])
       
        if choix_df == 'df1':
            
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 & option_2) :
                df1_option1_et_2 = df1.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['Player Name','Period','Action Type','Shot Type','Shot Zone Basic','Shot Zone Area','Shot Zone Range','Shot Distance','X Location','Y Location','Shot Made Flag']).set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['Game ID','Player ID','Team ID','Minutes Remaining','Seconds Remaining','Game Date','Team Name','Home Team','Away Team'])
                st.dataframe(df1_option1_et_2)
                
            elif option_1 :
                df1_option_1 = df1.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['Player Name','Period','Action Type','Shot Type','Shot Zone Basic','Shot Zone Area','Shot Zone Range','Shot Distance','X Location','Y Location','Shot Made Flag'])
                st.dataframe(df1_option_1)
                
            elif option_2 :
                df1_option_2 = df1.style.set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['Game ID','Player ID','Team ID','Minutes Remaining','Seconds Remaining','Game Date','Team Name','Home Team','Away Team'])
                st.dataframe(df1_option_2)
            
            else :
                st.dataframe(df1)
                  
                                                            # dataset 2                                     
            
    elif choix_dataset == 'dataset_2':
        choix_df = st.radio('Sélectionner le df d intérêt',['df00_01'])
        
        if choix_df == 'df00_01':
            
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 & option_2) :
                df00_01_option1_et_2 = df00_01.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['SCOREMARGIN','PERIOD']).set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['EVENTNUM','GAME_ID','PCTIMESTRING','PLAYER1_ID','PLAYER1_TEAM_ID'])
                st.dataframe(df00_01_option1_et_2)
                
            elif option_1 :
                df00_01_option_1 = df00_01.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['SCOREMARGIN','PERIOD'])
                st.dataframe(df00_01_option_1)
                
            elif option_2 :
                df00_01_option_2 = df00_01.style.set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['EVENTNUM','GAME_ID','PCTIMESTRING','PLAYER1_ID','PLAYER1_TEAM_ID'])
                st.dataframe(df00_01_option_2)
            
            else :
                st.dataframe(df00_01)
                 
                    
                                                          # dataset 3
            
            
    elif choix_dataset == 'dataset_3':
        choix_df = st.radio('Sélectionner le df d intérêt',['df4','df5','df6','df7','df8'])
        if choix_df == 'df4':
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            if (option_1 | option_2):
                st.write('Pas de variables pertinentes pour la suite')
            else :
                st.dataframe(df4)
            
        elif choix_df == 'df5':
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            if (option_1 | option_2):
                st.write('Pas de variables pertinentes pour la suite')
            else :
                st.dataframe(df5)
            
        elif choix_df == 'df6':
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 | option_2) :
                st.write('Pas de variables pertinentes pour la suite')
            else:
                st.dataframe(df6)
                
        elif choix_df == 'df7':
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 | option_2) :
                st.write('Pas de variables pertinentes pour la suite')
            else: 
                st.dataframe(df7)
            
        else:
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 & option_2) :
                df8_option1_et_2 = df8.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['W_PCT']).set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['TEAM_ID','SEASON_ID'])
                st.dataframe(df8_option1_et_2)
                
            elif option_1 :
                df8_option_1 = df8.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['W_PCT'])
                st.dataframe(df8_option_1)
                
            elif option_2 :
                df8_option_2 = df8.style.set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['TEAM_ID','SEASON_ID'])
                st.dataframe(df8_option_2)
            
            else :
                st.dataframe(df8)
            
            
            
                                                            # dataset 4
                
                
    elif choix_dataset == 'dataset_4':
        choix_df = st.radio('Sélectionner le df d intérêt',['df9','df10','df11'])
        if choix_df == 'df9':
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 & option_2) :
                df9_option1_et_2 = df9.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['birth_date','position']).set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['name'])
                st.dataframe(df9_option1_et_2)
                
            elif option_1 :
                df9_option_1 = df9.style.set_properties(**{'background-color' : 'firebrick','color' : 'white'}, subset = ['birth_date','position'])
                st.dataframe(df9_option_1)
                
            elif option_2 :
                df9_option_2 = df9.style.set_properties(**{'background-color' : 'royalblue','color' : 'white'}, subset = ['name'])
                st.dataframe(df9_option_2)
            
            else :
                st.dataframe(df9)
                
        elif choix_df == 'df10':
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 | option_2) :
                st.write('Pas de variables pertinentes pour la suite')
            else: 
                st.dataframe(df10)
            
        else :
            option_1 = st.checkbox(':red_circle: Montrer les variables gardées pour les parties dataviz et ML :red_circle:')
            option_2 = st.checkbox(':large_blue_circle: Montrer les variables gardées pour appliquer un merge entre les df ou pour créer de nouvelles variables :large_blue_circle:')
            
            if (option_1 | option_2) :
                st.write('Pas de variables pertinentes pour la suite')
            else: 
                st.dataframe(df11)
    
                                    # Partie 3 : Premières visualisations globales des données
    
    
if choix_partie == "III - Data visualization":   

    st.subheader("III - Data visualization")
    
    type_graphique = ['Distribution','Relation entre deux variables','Classement joueur ou poste','Joint Plot']
    graphique_choisi = st.radio("Quel est le type d'analyse graphique souhaitée?", type_graphique)

    # distribution
    if graphique_choisi == 'Distribution':
        features_list = list(df_viz.columns)
        x_choisi = st.selectbox(label = 'Choisi une variable en abscisse', options = features_list)
        if x_choisi in ['x_location','y_location','shot_distance','angle_tir','w_pct_adverse','score_margin','temps_restant']:
            nb_bins = st.number_input('Nombre de séparations sur le graphique',5,50)
            if st.checkbox('En pourcentage?'):
                ax = sns.displot(df_viz[x_choisi], bins = nb_bins, stat = 'percent', aspect = 3)
                plt.title('Distribution de la variable {}'.format(x_choisi))
                st.pyplot(plt)
            else:
                ax = sns.displot(df_viz[x_choisi], bins = nb_bins, aspect = 3)
                plt.title('Distribution de la variable {}'.format(x_choisi))
                st.pyplot(plt)
               
        else :
            if st.checkbox('Hue option?'):
                variable_hue = 'shot_made_flag'
                if st.checkbox('En pourcentage?'):
                    fig = plt.figure(figsize=(10,4))
                    ax = fig.add_subplot(111)
                    graph = df_viz.groupby(by = x_choisi)['shot_made_flag'].value_counts(normalize = True).mul(100).rename('percent').reset_index()
                    ax = sns.barplot(x = graph[x_choisi], y = graph['percent'], hue = graph['shot_made_flag'])
                    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
                    plt.bar_label(ax.containers[0], label_type = 'edge', fmt = '%.1f',fontsize = 5.5)
                    plt.bar_label(ax.containers[1], label_type = 'edge', fmt = '%.1f',fontsize = 5.5)
                    plt.title('Distribution de la variable {} en fonction de la réussite au tir'.format(x_choisi, variable_hue))
                    plt.legend(title = variable_hue,bbox_to_anchor=(1.0, 1.0), loc='upper left')
                    st.pyplot(plt)
                    
                else:
                    fig = plt.figure(figsize=(10,4))
                    ax = fig.add_subplot(111)
                    ax = sns.countplot(df_viz[x_choisi], hue = df_viz[variable_hue])
                    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
                    plt.bar_label(ax.containers[0], label_type = 'edge', fontsize = 4.5)
                    plt.bar_label(ax.containers[1], label_type = 'edge', fontsize = 4.5)
                    plt.title('Distribution de la variable {} en fonction de la réussite au tir'.format(x_choisi, variable_hue))
                    plt.legend(title = variable_hue,bbox_to_anchor=(1.0, 1.0), loc='upper left')
                    st.pyplot(plt)
            else :
                if st.checkbox('En pourcentage?'):
                    fig = plt.figure(figsize=(10,4))
                    ax = fig.add_subplot(111)
                    graph = df_viz[x_choisi].value_counts(normalize = True).mul(100).reset_index().rename(columns = {'{}'.format(x_choisi) : 'percent', 'index' : '{}'.format(x_choisi)})
                    ax = sns.barplot(x = graph[x_choisi], y = graph['percent'])
                    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
                    plt.bar_label(ax.containers[0], label_type = 'edge', fmt = '%.2f',fontsize = 8)
                    plt.title('Distribution de la variable {}'.format(x_choisi))
                    st.pyplot(plt)
                else:
                    fig = plt.figure(figsize=(10,4))
                    ax = fig.add_subplot(111)
                    ax = sns.countplot(df_viz[x_choisi], order = df_viz[x_choisi].value_counts().reset_index().sort_values(by = x_choisi, ascending = False)["index"])
                    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
                    plt.bar_label(ax.containers[0], label_type = 'edge', fontsize = 7)
                    plt.title('Distribution de la variable {}'.format(x_choisi))
                    st.pyplot(plt)
                

    # relation entre deux variables
    if graphique_choisi == 'Relation entre deux variables':
        features_list = ['shot_distance','x_location','y_location','angle_tir']
        x_choisi = st.selectbox(label = 'Choisi une variable en abscisse', options = list(df_viz.columns))
        list_1 = list(df_viz.columns)
        list_2 = []
        for i in list_1:
            if i != x_choisi:
                list_2.append(i)
        if x_choisi in features_list:        
            y_choisi = st.selectbox(label = 'Choisi une variable en ordonnée', options = list(df_viz.columns))
        else:
            y_choisi = st.selectbox(label = 'Choisi une variable en ordonnée', options = features_list)
                                                                                             
        if (x_choisi in features_list) & (y_choisi in features_list):
            if st.checkbox('Hue option?'):
                variable_hue = 'shot_made_flag'
                ax = sns.relplot(data = df_viz, x = x_choisi, y = y_choisi, hue = variable_hue, aspect = 2)
                plt.title('Relation entre {} et {}'.format(x_choisi,y_choisi), fontdict = {'fontsize' : 20})
                ax.set_xticklabels(rotation = 90)
                st.pyplot(plt)
            else:
                ax = sns.relplot(data = df_viz, x = x_choisi, y = y_choisi, aspect = 2.5)
                plt.title('Relation entre {} et {}'.format(x_choisi,y_choisi), fontdict = {'fontsize' : 20})
                ax.set_xticklabels(rotation = 90)
                st.pyplot(plt)
        else :
            if st.checkbox('Hue option?'):
                variable_hue = 'shot_made_flag'
                ax = sns.catplot(data = df_viz, x = x_choisi, y = y_choisi,hue = variable_hue, kind = st.selectbox(label = 'Type de graphique', options = ['box','violin','boxen']), aspect = 3)
                plt.title('{} en fonction de {}'.format(y_choisi,x_choisi),fontdict = {'fontsize' : 20})
                ax.set_xticklabels(rotation = 90)
                plt.ylabel('{}'.format(y_choisi), fontsize=15)
                plt.xlabel('{}'.format(x_choisi), fontsize=15)
                st.pyplot(plt)
            else:

                ax = sns.catplot(data = df_viz, x = x_choisi, y = y_choisi, kind = st.selectbox(label = 'Type de graphique', options = ['box','violin','boxen']), aspect = 3)
                plt.title('{} en fonction de {}'.format(y_choisi,x_choisi),fontdict = {'fontsize' : 20})
                ax.set_xticklabels(rotation = 90)
                plt.ylabel('{}'.format(y_choisi), fontsize=15)
                plt.xlabel('{}'.format(x_choisi), fontsize=15)
                st.pyplot(plt)



    # classement joueur ou poste par zone
    if graphique_choisi == 'Classement joueur ou poste':
        features_list = list(df_viz.columns)
        x_choisi = st.selectbox(label = 'Choisi une variable en abscisse', options = ['player_name','position'])
        nb_var_x = st.slider("Nb de données à montrer",min_value = 1,max_value = df_viz[x_choisi].unique().size,value = df_viz[x_choisi].unique().size)
        y_choisi = 'shot_made_flag'
        zone_1 = st.selectbox(label = 'Choisi une variable zone d intérêt', options = ['action_type', 'shot_type', 'shot_zone_basic', 'shot_zone_area','shot_zone_range'])
        with st.expander('Cliquez ici pour choisir une variable plus fine'):
            zone_choisi = st.radio('Parmi cette variable, quelle zone souhaites tu étudier?',list(df_viz[zone_1].unique()))

        
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        graph = pd.DataFrame(df_viz[df_viz[zone_1] == zone_choisi].groupby(by = x_choisi)['shot_made_flag'].mean().reset_index().sort_values(by = 'shot_made_flag',ascending = False).head(nb_var_x))
        ax = sns.barplot(x = x_choisi, y = 'shot_made_flag', data = graph)
        plt.xticks(rotation = 90)
        plt.ylim(0,1)
        plt.bar_label(ax.containers[0],fmt = '%.3f', label_type = 'edge', fontsize = 7)
        plt.title('TOP {} {} {}'.format(nb_var_x,x_choisi,zone_choisi));
        st.pyplot(plt);
        

    #Création du court

    

    
    #Plot des jointplot avec une boucle sur tous les joueurs
    if graphique_choisi == 'Joint Plot':
        player_name_list = list(df_viz['player_name'].unique())
        player_choisi = st.selectbox(label = 'Choisi un joueur parmi ceux proposés?', options = player_name_list)
        @st.cache_data
        def joint_plot(player_choisi):
            df_viz_player = df_viz[df_viz['player_name'] == player_choisi]
            cmap = plt.cm.jet
            plot = sns.jointplot(x = df_viz_player['x_location'],y= df_viz_player['y_location'],kind='kde',cmap=cmap,n_levels =100,fill = True,joint_kws={'thresh':0.})
            ax = plot.ax_joint
            ax.grid(False)
            plot.fig.set_size_inches(4,3.76)
            #fig, axs = plt.subplots(1,2, figsize=(4*4, 2*3.76))
            ax.set_xlim(-250, 250)
            ax.set_ylim(0, 470)
            ax.tick_params(labelbottom='False', labelleft='False')
            draw_court(ax=ax, color='white', lw=2, outer_lines=True)
            plot.ax_marg_x.grid(False)
            plt.title('{}'.format(player_choisi),y=-0.1,fontsize=10)
            plt.grid(False)
            st.pyplot(plt)
        joint_plot(player_choisi)

                            # Partie 4 : Analyse des tirs de 20 des meilleurs joueurs de NBA du 21ème siècle
    
if choix_partie == "IV - Analyse des tirs de 20 des meilleurs joueurs de NBA du 21ème siècle":
    
    st.subheader('IV - Analyse des tirs de 20 des meilleurs joueurs de NBA du 21ème siècle') 


    
    
                                        # choix des paramètres pour la visualisation

                                                    # player_name
    col1,col2 = st.columns(2)
    
    with col1:
        with st.expander('Choisis le joueur n°1'):
            choix_player_1 = st.radio('Player_Name_1',list(np.append('All',df_viz['player_name'].unique())))
    with col2:
        with st.expander('Choisis le joueur n°2'):
            choix_player_2 = st.radio('Player_Name_2',list(np.append('All',df_viz['player_name'].unique())))
    choix_parametres = st.multiselect('Quel(s) paramètres à faire varier pour la comparaison?', ['year_game_date','shot_type','action_type','w_pct_adverse','classement_adversaire','period','temps_restant','shot_zone_basic','shot_zone_area','shot_zone_range','shot_distance','angle_tir','position','domicile','score_margin','mène'])
    nb_parametres = len(choix_parametres)
    liste_parametres_bornes = ['w_pct_adverse','classement_adversaire','temps_restant','shot_distance','angle_tir','score_margin']
  
    
    
    

    
                                                         #year
               
    def widget_parametres(parametre):
        
        for i in range(len(choix_parametres)):
            col1,col2 = st.columns(2)
                
            if choix_parametres[i] in liste_parametres_bornes :
                with col1:
                
                    if st.checkbox('Including_all_{}_player_1?'.format(parametre[i])):
                        globals()[str('choix_') + choix_parametres[i] + str('_1')] = 'All'
                    else:
                        globals()[str('choix_') + choix_parametres[i] + str('_1')] = st.slider('{}_1'.format(parametre[i]),df_viz['{}'.format(parametre[i])].min(),df_viz['{}'.format(parametre[i])].max(),(df_viz['{}'.format(parametre[i])].min(),df_viz['{}'.format(parametre[i])].max()))
                with col2:
                
                    if st.checkbox('Including_all_{}_player_2?'.format(parametre[i])):
                        globals()[str('choix_') + choix_parametres[i] + str('_2')] = 'All'
                    else:
                        globals()[str('choix_') + choix_parametres[i] + str('_2')] = st.slider('{}_2'.format(parametre[i]),df_viz['{}'.format(parametre[i])].min(),df_viz['{}'.format(parametre[i])].max(),(df_viz['{}'.format(parametre[i])].min(),df_viz['{}'.format(parametre[i])].max()))
                        
            elif df_viz['{}'.format(parametre[i])].dtype == int:
                col1,col2 = st.columns(2)
                
                with col1:
                                             
                    if parametre[i] in choix_parametres:
                        if st.checkbox('Including_all_{}_player_1?'.format(parametre[i])):
                            globals()[str('choix_') + choix_parametres[i] + str('_1')] = 'All'
                        elif choix_player_1 == 'All':
                            globals()[str('choix_') + choix_parametres[i] + str('_1')] = st.slider('{}_1'.format(parametre[i]),df_viz['{}'.format(parametre[i])].min(),df_viz['{}'.format(parametre[i])].max())
                        else:
                            globals()[str('choix_') + choix_parametres[i] + str('_1')] = st.slider('{}_1'.format(parametre[i]),df_viz[df_viz['player_name'] == choix_player_1]['{}'.format(parametre[i])].min(),df_viz[df_viz['player_name'] == choix_player_1]['{}'.format(parametre[i])].max())
                    else:
                        globals()[str('choix_') + choix_parametres[i] + str('_1')] = 'All'      
                
                with col2:
                    if parametre[i] in choix_parametres:
                        if st.checkbox('Including_all_{}_player_2?'.format(parametre[i])):
                            globals()[str('choix_') + choix_parametres[i] + str('_2')] = 'All'
                        elif choix_player_2 == 'All':
                            globals()[str('choix_') + choix_parametres[i] + str('_2')] = st.slider('{}_2'.format(parametre[i]),df_viz['{}'.format(parametre[i])].min(),df_viz['{}'.format(parametre[i])].max())
                        else:
                            globals()[str('choix_') + choix_parametres[i] + str('_2')] = st.slider('{}_2'.format(parametre[i]),df_viz[df_viz['player_name'] == choix_player_2]['{}'.format(parametre[i])].min(),df_viz[df_viz['player_name'] == choix_player_2]['{}'.format(parametre[i])].max())
                    else:
                        globals()[str('choix_') + choix_parametres[i] + str('_2')] = 'All'    
                        
                        
                        
            else:        
                col1,col2 = st.columns(2)   

                with col1:
                    if parametre[i] in choix_parametres:
                        if st.checkbox('Including_all_{}_player_1?'.format(parametre[i])):
                            globals()[str('choix_') + choix_parametres[i] + str('_1')] = 'All'
                        elif choix_player_1 == 'All':
                            globals()[str('choix_') + choix_parametres[i] + str('_1')] = st.selectbox('{}_1'.format(parametre[i]),list(df_viz['{}'.format(parametre[i])].value_counts().reset_index()['index']))
                        else:
                            globals()[str('choix_') + choix_parametres[i] + str('_1')] = st.selectbox('{}_1'.format(parametre[i]),list(df_viz[df_viz['player_name'] == choix_player_1]['{}'.format(parametre[i])].value_counts().reset_index()['index']))
                    else:
                        globals()[str('choix_') + choix_parametres[i] + str('_1')] = 'All'

                with col2:
                    if parametre[i] in choix_parametres:
                        if st.checkbox('Including_all_{}_player_2?'.format(parametre[i])):
                            globals()[str('choix_') + choix_parametres[i] + str('_2')] = 'All'
                        elif choix_player_2 == 'All':
                            globals()[str('choix_') + choix_parametres[i] + str('_2')] = st.selectbox('{}_2'.format(parametre[i]),list(df_viz['{}'.format(parametre[i])].value_counts().reset_index()['index']))
                        else:
                            globals()[str('choix_') + choix_parametres[i] + str('_2')] = st.selectbox('{}_2'.format(parametre[i]),list(df_viz[df_viz['player_name'] == choix_player_2]['{}'.format(parametre[i])].value_counts().reset_index()['index']))
                    else:
                        globals()[str('choix_') + choix_parametres[i] + str('_2')] = 'All'
                        return globals()[str('choix_') + choix_parametres[i] + str('_1')],globals()[str('choix_') + choix_parametres[i] + str('_2')]

    widget_parametres(choix_parametres)
    
        
    def update_plot(choix_player_1,choix_player_2, nb_parametres):
                    if nb_parametres == 0:
                        if choix_player_1 == 'All':
                            player_1_data = df_viz
                        else:
                            player_1_data = df_viz[df_viz['player_name'] == choix_player_1]
                        if choix_player_2 == 'All':
                            player_2_data = df_viz
                        else:
                            player_2_data = df_viz[df_viz['player_name'] == choix_player_2]

                    elif nb_parametres == 1:
                        if choix_player_1 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All':
                                player_1_data = df_viz
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1])]
                                else:                                  
                                    player_1_data = df_viz[df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')]]
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All':
                                player_1_data = df_viz[df_viz['player_name'] == choix_player_1]
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1]) & (df_viz['player_name'] == choix_player_1)]
                                else:
                                    player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')]) & (df_viz['player_name'] == choix_player_1)]
                        if choix_player_2 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                player_2_data = df_viz
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_2')][1])]
                                else:                                  
                                    player_2_data = df_viz[df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')]]
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                player_2_data = df_viz[df_viz['player_name'] == choix_player_2]
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1]) & (df_viz['player_name'] == choix_player_2)]
                                else:
                                    player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')]) & (df_viz['player_name'] == choix_player_2)]
                            
                           
                    else:
                        if choix_player_1 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    player_1_data = df_viz
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_1')][1])]
                                    else:
                                        player_1_data = df_viz[df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_1')]]
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    if choix_parametres[0] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1])]
                                    else:
                                        player_1_data = df_viz[df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')]]
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1]) & (df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_1')][1])]
                                    
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_1')])]
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_1')][1]) & (df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')])]
                                    else:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_1')])]
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    player_1_data = df_viz[df_viz['player_name'] == choix_player_1]
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_1')][1]) & (df_viz['player_name'] == choix_player_1)]
                                    else:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_1')]) & (df_viz['player_name'] == choix_player_1)]
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    if choix_parametres[0] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1]) & (df_viz['player_name'] == choix_player_1)]
                                    else:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')]) & (df_viz['player_name'] == choix_player_1)]
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1]) & (df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_1')][1]) & (df_viz['player_name'] == choix_player_1)]
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_1')][1]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_1')]) & (df_viz['player_name'] == choix_player_1)]
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_1')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_1')][1]) & (df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')]) & (df_viz['player_name'] == choix_player_1)]
                                    
                                    else:
                                        player_1_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_1')]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_1')]) & (df_viz['player_name'] == choix_player_1)]


                        if choix_player_2 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    player_2_data = df_viz
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_2')][1])]
                                    else:
                                        player_2_data = df_viz[df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_2')]]
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    if choix_parametres[0] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_2')][1])]
                                    else:
                                        player_2_data = df_viz[df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')]]
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_2')][1]) & (df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_2')][1])]
                                        
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_2')][1]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_2')])]
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_2')][1]) & (df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')])]
                                   
                                    else:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_2')])]
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    player_2_data = df_viz[df_viz['player_name'] == choix_player_2]
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_2')][1]) & (df_viz['player_name'] == choix_player_2)]
                                    else:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_2')]) & (df_viz['player_name'] == choix_player_2)]
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    if choix_parametres[0] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_2')][1]) & (df_viz['player_name'] == choix_player_2)]
                                    else:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')]) & (df_viz['player_name'] == choix_player_2)]
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_2')][1]) & (df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_2')][1]) & (df_viz['player_name'] == choix_player_2)]
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] >= globals()[str('choix_') + choix_parametres[0] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[0])] <= globals()[str('choix_') + choix_parametres[0] + str('_2')][1]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_2')]) & (df_viz['player_name'] == choix_player_2)]
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[1])] >= globals()[str('choix_') + choix_parametres[1] + str('_2')][0]) & (df_viz['{}'.format(choix_parametres[1])] <= globals()[str('choix_') + choix_parametres[1] + str('_2')][1]) & (df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')]) & (df_viz['player_name'] == choix_player_2)]
                                   
                                    else:
                                        player_2_data = df_viz[(df_viz['{}'.format(choix_parametres[0])] == globals()[str('choix_') + choix_parametres[0] + str('_2')]) & (df_viz['{}'.format(choix_parametres[1])] == globals()[str('choix_') + choix_parametres[1] + str('_2')]) & (df_viz['player_name'] == choix_player_2)]
            
                
                
                
                
                    gridsize = 50
                    plt.figure(figsize=(10,10))
                    shot_df_1 = player_1_data
                    shot_df_2 = player_2_data
                    cmap=LinearSegmentedColormap.from_list('rg',["red", "yellow","lawngreen"], N=256)  

                    ######################## graphique de gauche #########################
                    x_2pts = shot_df_1['x_location']
                    y_2pts = shot_df_1['y_location']
                    x_2pts_made = shot_df_1['x_location'][(shot_df_1['shot_made_flag']==1)]
                    y_2pts_made = shot_df_1['y_location'][(shot_df_1['shot_made_flag']==1)]

                    ##### Nombre de shots pris et réussis pour chaque hexagone
                    hexa_shot2pts = plt.hexbin(x_2pts, y_2pts, gridsize=gridsize, extent=(-250,250,425,-50),cmap=cmap);
                    hexa_made2pts = plt.hexbin(x_2pts_made, y_2pts_made, gridsize=gridsize, extent=(-250,250,425,-50),cmap=cmap);

                    #### Calcul du pourcentage de réussite
                    Shot_2pts_made_percentage = hexa_made2pts.get_array() / hexa_shot2pts.get_array()
                    Shot_2pts_made_percentage[np.isnan(Shot_2pts_made_percentage)] = 0 #transforme les Nan en 0
                    OFFSETS_2pts = hexa_shot2pts.get_offsets()
                    #######################################################


                    ######################## graphique de droite #########################
                    x_3pts = shot_df_2['x_location']
                    y_3pts = shot_df_2['y_location']
                    x_3pts_made = shot_df_2['x_location'][(shot_df_2['shot_made_flag']==1)]
                    y_3pts_made = shot_df_2['y_location'][(shot_df_2['shot_made_flag']==1)]

                    ##### Nombre de shots pris et réussis pour chaque hexagone
                    hexa_shot3pts = plt.hexbin(x_3pts, y_3pts, gridsize=gridsize, extent=(-250,250,425,-50),cmap=cmap);
                    hexa_made3pts = plt.hexbin(x_3pts_made, y_3pts_made, gridsize=gridsize, extent=(-250,250,425,-50),cmap=cmap);

                    #### Calcul du pourcentage de réussite
                    Shot_3pts_made_percentage = hexa_made3pts.get_array() / hexa_shot3pts.get_array()
                    Shot_3pts_made_percentage[np.isnan(Shot_3pts_made_percentage)] = 0 #transforme les Nan en 0
                    OFFSETS_3pts = hexa_shot3pts.get_offsets()
                    #######################################################

                    ###### Création Figure de gauche
                    fig, axs = plt.subplots(1, 2,figsize=(20,6))

                    for i in np.arange(len(hexa_shot2pts.get_array())):
                        if hexa_shot2pts.get_array()[i] > 240/gridsize: 
                            hexa_shot2pts.get_array()[i] = 240/gridsize

                    sc = axs[0].scatter(OFFSETS_2pts[:, 0], OFFSETS_2pts[:, 1], c = Shot_2pts_made_percentage, s = hexa_shot2pts.get_array()*10,cmap=cmap, marker='h')

                    ##### Ajout du terrain
                    axs[0] = draw_court(ax = axs[0],color='black',outer_lines=True)
                    axs[0].set_xlim(-260,260)
                    axs[0].set_ylim(-50, 425)
                    #####

                    ##### Ajout colorbar
                    cb_2pts = fig.colorbar(sc,ax = axs[0], orientation='vertical')
                    cb_2pts.set_label('% réussite au shoot',labelpad=0,fontsize = 20)
                    cb_2pts.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
                    cb_2pts.set_ticklabels(['0%','25%', '50%','75%', '100%'])

                    ####### Ajout de la légende
                    shots_by_hex_2pts = hexa_shot2pts.get_array()
                    freq_by_hex_2pts = shots_by_hex_2pts / sum(shots_by_hex_2pts)
                    sizes_2pts = freq_by_hex_2pts
                    sizes_2pts = sizes_2pts / max(sizes_2pts)
                    max_freq_2pts = max(freq_by_hex_2pts)
                    max_size_2pts = max(sizes_2pts)

                    legend_2pts = axs[0].legend(*sc.legend_elements('sizes', num=6,alpha=0.5, fmt="{x:.2f}%"
                                                                , func=lambda s: (s / max_size_2pts) * max_freq_2pts * 10),
                                                                loc='upper left', title='Freq (%)', fontsize='small',
                                                                facecolor='white',framealpha=0.5)

                    ##### Titre figure de gauche
                    if nb_parametres == 0:
                        if choix_player_1 == 'All':
                            axs[0].set_title('All players shoots', fontsize = 30)
                        else:
                            axs[0].set_title(shot_df_1['player_name'].unique()[0] + ' shoots', fontsize = 30)
                            
                    elif nb_parametres == 1:
                        if choix_player_1 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All': 
                                axs[0].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')],choix_parametres[0]), fontsize = 30)
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    axs[0].set_title('All players shoots ' + choix_parametres[0] + ' :' + '  {}-{}'.format(np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)), fontsize = 30)
                                else:
                                    axs[0].set_title('All players shoots' + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')]), fontsize = 30)
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All':
                                axs[0].set_title(shot_df_1['player_name'].unique()[0]+ ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')],choix_parametres[0]), fontsize = 30)
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    axs[0].set_title(shot_df_1['player_name'].unique()[0] + ' : ' + choix_parametres[0] + ' : ' + '{}-{}'.format(np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)), fontsize = 30)
                                    
                                else:
                                    axs[0].set_title(shot_df_1['player_name'].unique()[0] + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')]), fontsize = 30)
                            
                    else:
                        if choix_player_1 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    axs[0].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')],choix_parametres[0]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        axs[0].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_1')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][1],2)), fontsize = 30)
                                    else:
                                        axs[0].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_1')],choix_parametres[0]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')]), fontsize = 30)
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    if choix_parametres[0] in liste_parametres_bornes:
                                        axs[0].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_1')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)), fontsize = 30)
                                    else:
                                        axs[0].set_title('All players shoots' + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        axs[0].set_title('All players shoots' + ' - {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][1],2)), fontsize = 30)
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        axs[0].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_1')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)), fontsize = 30)
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        axs[0].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_1')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][1],2)), fontsize = 30)
                                    
                                    else:
                                        axs[0].set_title('All players shoots' + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')]), fontsize = 30)
                                                     
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_1')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    axs[0].set_title(shot_df_1['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')],choix_parametres[0]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_1')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][1],2)), fontsize = 30)
                                    else:
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_1')],choix_parametres[0]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')]), fontsize = 30)
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_1')] == 'All':
                                    if choix_parametres[0] in liste_parametres_bornes:
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_1')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)), fontsize = 30)
                                    else:
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0]  + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0] + ' - {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][1],2)), fontsize = 30)
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_1')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_1')][1],2)), fontsize = 30)
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0] + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_1')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_1')][1],2)), fontsize = 30)
                                    
                                    else:
                                        axs[0].set_title(shot_df_1['player_name'].unique()[0] + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_1')]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_1')]), fontsize = 30)
                                                     
                                                     



                    ###### Création Figure de droite

                    for i in np.arange(len(hexa_shot3pts.get_array())):
                        if hexa_shot3pts.get_array()[i] > 240/gridsize: 
                            hexa_shot3pts.get_array()[i] = 240/gridsize

                    sc1 = axs[1].scatter(OFFSETS_3pts[:, 0], OFFSETS_3pts[:, 1], c = Shot_3pts_made_percentage, s = hexa_shot3pts.get_array()*10,cmap=cmap, marker='h')

                    ##### Ajout du terrain
                    axs[1] = draw_court(ax = axs[1],color='black',outer_lines=True)
                    axs[1].set_xlim(-260,260)
                    axs[1].set_ylim(-50, 425)
                    ##### Ajout colorbar 3PTS

                    cb_3pts = fig.colorbar(sc1,ax = axs[1], orientation='vertical')
                    cb_3pts.set_label('% réussite au shoot',labelpad=0,fontsize = 20)
                    cb_3pts.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
                    cb_3pts.set_ticklabels(['0%','25%', '50%','75%', '100%'])
                    ####### Ajout de la légende
                    shots_by_hex_3pts = hexa_shot3pts.get_array()
                    freq_by_hex_3pts = shots_by_hex_3pts / sum(shots_by_hex_3pts)
                    sizes_3pts = freq_by_hex_3pts
                    sizes_3pts = sizes_3pts / max(sizes_3pts)
                    max_freq_3pts = max(freq_by_hex_3pts)
                    max_size_3pts = max(sizes_3pts)

                    legend_3pts = axs[1].legend(*sc1.legend_elements('sizes', num = 6,alpha=0.5, fmt="{x:.2f}%"
                                                                , func=lambda s: (s / max_size_3pts) * max_freq_3pts*10),
                                                                loc='upper left', title='Freq (%)', fontsize='small',
                                                                facecolor='white',framealpha=0.5)

                    ##### Titre figure de droite
                    if nb_parametres == 0:
                        if choix_player_2 == 'All':
                            axs[1].set_title('All players shoots', fontsize = 30)
                        else:
                            axs[1].set_title(shot_df_2['player_name'].unique()[0] + ' shoots', fontsize = 30)
                            
                    elif nb_parametres == 1:
                        if choix_player_2 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                axs[1].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')],choix_parametres[0]), fontsize = 30)
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    axs[1].set_title('All players shoots ' + choix_parametres[0] + ' :' + '  {}-{}'.format(np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)), fontsize = 30)
                                else:
                                    axs[1].set_title('All players shoots' + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')]), fontsize = 30)
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                axs[1].set_title(shot_df_2['player_name'].unique()[0]+ ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')],choix_parametres[0]), fontsize = 30)
                            else:
                                if choix_parametres[0] in liste_parametres_bornes:
                                    axs[1].set_title(shot_df_2['player_name'].unique()[0] + ' : ' + choix_parametres[0] + ' : ' + '{}-{}'.format(np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)), fontsize = 30)
                                else:
                                    axs[1].set_title(shot_df_2['player_name'].unique()[0] + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')]), fontsize = 30)
                            
                    else:
                        if choix_player_2 == 'All':
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    axs[1].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')],choix_parametres[0]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        axs[1].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_2')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][1],2)), fontsize = 30)
                                    else:
                                        axs[1].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_2')],choix_parametres[0]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')]), fontsize = 30)
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        axs[1].set_title('All players shoots' + ' - {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][1],2)), fontsize = 30)
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        axs[1].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_2')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)), fontsize = 30)
                                    else:
                                        axs[1].set_title('All players shoots' + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        axs[1].set_title('All players shoots' + ' - {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][1],2)), fontsize = 30)
                                        
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        axs[1].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_2')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)), fontsize = 30)
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        axs[1].set_title('All players shoots' + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_2')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][1],2)), fontsize = 30)
                                    
                                    else:
                                        axs[1].set_title('All players shoots' + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')]), fontsize = 30)
                                                     
                        else:
                            if globals()[str('choix_') + choix_parametres[0] + str('_2')] == 'All':
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    axs[1].set_title(shot_df_2['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')],choix_parametres[0]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if choix_parametres[1] in liste_parametres_bornes:
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_2')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][1],2)), fontsize = 30)
                                    else:
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_2')],choix_parametres[0]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')]), fontsize = 30)
                            else:
                                if globals()[str('choix_') + choix_parametres[1] + str('_2')] == 'All':
                                    if choix_parametres[0] in liste_parametres_bornes:
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_2')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)), fontsize = 30)
                                    else:
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0]  + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')]) + ' \n {} {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')],choix_parametres[1]), fontsize = 30)
                                else:
                                    if (choix_parametres[0] in liste_parametres_bornes and choix_parametres[1] in liste_parametres_bornes):
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0] + ' - {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][1],2)), fontsize = 30)
                                    elif choix_parametres[0] in liste_parametres_bornes:
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0]  + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[1]+ str('_2')],choix_parametres[1]) + ' \n {} : {}-{}'.format(choix_parametres[0],np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[0] + str('_2')][1],2)), fontsize = 30)
                                    elif choix_parametres[1] in liste_parametres_bornes:
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0] + ' - {} {}'.format(globals()[str('choix_')+ choix_parametres[0]+ str('_2')],choix_parametres[0]) + ' \n {} : {}-{}'.format(choix_parametres[1],np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][0],2),np.round(globals()[str('choix_') + choix_parametres[1] + str('_2')][1],2)), fontsize = 30)
                                    
                                    else:
                                        axs[1].set_title(shot_df_2['player_name'].unique()[0] + ' - {}'.format(globals()[str('choix_') + choix_parametres[0] + str('_2')]) + ' \n {}'.format(globals()[str('choix_') + choix_parametres[1] + str('_2')]), fontsize = 30)
                    
                    return st.pyplot(plt)
                        

    update_plot(choix_player_1,choix_player_2, nb_parametres)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                    # Partie 5 : Modèle de prédiction des tirs de joueurs de NBA
if choix_partie == "V - Modèle de prédiction des tirs de joueurs de NBA" :
    
    st.subheader('V - Modèle de prédiction des tirs de joueurs de NBA')    


    # préparation des données et séparation des données de tests et d'entraînement
    with st.expander('Visualiser la préparation des données'):
        tab1,tab2 = st.tabs(['Machine Learning','Deep Learning'])
        with tab1:
            st.code('''
#encodage des variables catégorielles            
df = pd.get_dummies(data = df)

# séparation des données en features et target
target = df['shot_made_flag']
features = df.drop('shot_made_flag', axis = 1)

# séparation en un ensemble d'entraînement et un ensemble de test
x_train,x_test,y_train,y_test = train_test_split(features,target, test_size = 0.2, random_state = np.random)

# standardisation des données (x_train et x_test)
scaler = preprocessing.StandardScaler().fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)''', language = 'python')
        with tab2:
            st.code('''
#encodage des variables catégorielles            
df = pd.get_dummies(data = df)

# séparation des données en features et target
target = df['shot_made_flag']
features = df.drop('shot_made_flag', axis = 1)

# séparation en un ensemble d'entraînement et un ensemble de test
x_train,x_test,y_train,y_test = train_test_split(features,target, test_size = 0.2, random_state = np.random)

# standardisation des données (x_train et x_test)
scaler = preprocessing.StandardScaler().fit(x_train)''', language = 'python')

    # liste des modèles

    with st.expander('Visualiser les différents modèles testés'):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['Logistic_Regression','Decision_Tree','Random_Forest','XGBoost', 'Deep_Learning_RNN'])
        with tab1:
            st.code('''#recherche d'hyperparamètres
clf_lr = linear_model.LogisticRegression()

parametres = {'C' : [0.0001,0.001,0.01,0.1], # inverse de la régularisation (pénaliser des valeurs extrêmes pour réduire l'overfitting)
              'penalty' : ['l1', 'l2'], #fonction de pénalité
              'solver' : ['lbfgs','sag','saga']} #algorithme appliqué

grid_clf_lr = GridSearchCV(estimator = clf_lr,param_grid = parametres).fit(x_train_scaled,y_train)

#construction du modèle avec les hyperparamètres
linear_model.LogisticRegression(C = 0.0001, penalty = 'l2', solver = 'lbfgs')''', language = 'python')
        with tab2:
            st.code('''
#recherche d'hyperparamètres
clf_dt = DecisionTreeClassifier()

parametres = {'criterion' : ['gini','entropy','log_loss'], # fonction pour mesurer la qualité de la division d'un noeud
              'max_depth' : np.arange(4,16,2)} #profondeur maximale de l'arbre
              
grid_clf_dt = GridSearchCV(estimator = clf_dt,param_grid = parametres).fit(x_train_scaled,y_train)

#construction du modèle avec les hyperparamètres            
DecisionTreeClassifier(random_state = 123, criterion = 'entropy', max_depth = 10)''', language = 'python')
        with tab3:
            st.code('''
            
#recherche d'hyperparamètres
clf_rf = ensemble.RandomForestClassifier()

parametres = {'criterion' : ['gini','entropy','log_loss'], # fonction pour mesurer la qualité de la division d'un noeud
               'n_estimators = np.arange(200,1200,200)} #nombre d'arbres dans la forêt

grid_clf_rf = GridSearchCV(estimator = clf_rf,param_grid = parametres).fit(x_train_scaled,y_train)    

ensemble.RandomForestClassifier(random_state = 321, criterion = 'gini', n_estimators = 800, max_depth = None)''', language = 'python')
        with tab4:
            st.code('''xgb.XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_bin=256, max_cat_threshold=64, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
              monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0)''', language = 'python')
        with tab5:
            st.code('''import keras_tuner as kt

def model_builder(hp):
    input_model = Input(shape = x_train_scaled.shape[1])
    
    # les variables à faire varier
    hp.units_1 = hp.Int('units',min_value = 64, max_value = 512, step = 64)
    hp.units_2 = hp.Int('units',min_value = 64, max_value = 512, step = 64)
    hp_dropout = hp.Float('taux',min_value = 0.2, max_value = 0.6, step =0.2)
    hp_activation_dense = hp.Choice( "dense_activation", values=["relu", "tanh"])
    
    # les différentes couches
    first_layer = Dense(units = hp.units_1, activation = hp_activation_dense)
    second_layer= Dropout(rate = hp_dropout)
    third_layer = Dense(units = hp.units_2, activation = hp_activation_dense)
    fourth_layer= Dropout(rate = hp_dropout)
    fifth_layer = Dense(units = 1, activation = 'sigmoid')
    
    x = first_layer(input_model)
    x = second_layer(x)
    x = third_layer(x)
    x = fourth_layer(x)
    output_model = fifth_layer(x)
        
    model = Model(inputs = input_model, outputs = output_model)

    #callbacks
    early_stopping = callbacks.EarlyStopping(monitor = 'val_loss', verbose = 1, patience = 5,
                                         mode = 'min', restore_best_weights = True)
    lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, mode = 'min', verbose = 2)

    #compilation du modèle
    hp.lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp.lr),
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])

    #entraînement du modèle
    history = model.fit(x_train_scaled, y_train, batch_size = 200, epochs = 20, validation_split = 0.1,
                    validation_data = (x_test_scaled, y_test), callbacks = [early_stopping,lr_plateau]) 
    
    return model
    
tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy', 
                     max_epochs = 20,
                     factor = 4,
                     directory = 'my_dir',
                     project_name = 'hyper_tuning_DL_MSPy') 
                     
tuner.search(x_train_scaled, y_train, epochs=20, steps_per_epoch=20, validation_data = (x_test_scaled,y_test),
             verbose = 1, validation_steps=3, callbacks = [early_stopping,lr_plateau])''', language = 'python')
            
    
    with st.expander('Visualiser les scores des différents modèles'):
        col1, col2, col3, col4,col5 = st.columns(5)
        model_list = ['Logistic_Regression','Decision_Tree','Random_Forest','XGBoost', 'Deep_Learning_RNN']
        max_metric = 0
        for i in model_list:
            if np.load('score_{}.npy'.format(i))> max_metric:
                max_metric = np.load('score_{}.npy'.format(i))
            else:
                max_metric = max_metric
        for i in range(len(model_list)):
            with globals()[str('col') + '{}'.format(i + 1)]:
                img_1 = Image.open("confusion_matrix_{}.png".format(model_list[i]))
                st.image(img_1, caption="Matrice de confusion {}".format(model_list[i]))
                st.metric(label = 'Score test {}'.format(model_list[i]),value = np.round(np.load('score_{}.npy'.format(model_list[i])),3), delta = np.round(np.load('score_{}.npy'.format(model_list[i])) - max_metric,3))


    with st.expander('Modèle retenu et interprétabilité'):
        col1,col2 = st.columns(2)
        with col1:
            st.image(Image.open('summary_plot_bar.png'))
            
        with col2:
            st.image(Image.open('summary_plot_dot.png'))
        numero = st.slider('Numéro local',0,9)
        st.subheader(open('title_wrong_{}'.format(numero)).read())   
        col1,col2 = st.columns([2,1])
        with col1:
            st.image(Image.open('plot_wrong_{}.png'.format(numero)))
        with col2:
            st.dataframe(pd.read_pickle('df_wrong_{}.pkl'.format(numero)))
                                        # Partie 6 : Test du modèle sur de nouvelles données    
    
if choix_partie == "VI - Test du modèle sur de nouvelles données":
    
    st.markdown("### Et toi, vas-tu réussir à marquer ce prochain panier?")

    # caractéristiques du tir
    player_name = 'Reggie Bullock'
    equipe_joueur =  'Mavericks de Dallas'
    equipe_adverse = 'Warriors de Golden State'
    period = 2
    action_type = 'Jump Shot'
    shot_type = 3
    x_location = -235
    y_location = 40
    shot_distance = 7.0104
    poste = 'Arrière'
    position = 2
    age = 32
    w_pct_adverse = 0.514
    domicile = 1
    angle_tir = 80.340103
    temps_restant = 154
    score = '55 - 51'
    score_margin = -4
    mène = -1
    
    col1,col2 = st.columns(2)
    with col1:
        st.table(pd.DataFrame({'caractéristiques' : ['Reggie Bullock','Mavericks de Dallas','Warriors de Golden State',0.514,'GS - DAL','Arrière',32,'2ème quart-temps', '2:34','Jump Shot', '3_pts','55 - 51']}, 
             index = ['player_name','equipe_joueur','equipe_adverse','%PCT equipe_adverse','match','position','age','period','temps_restant','action_type','shot_type', 'score au moment du tir']))
    with col2:
        draw_court(color='black',outer_lines=True)
        plt.scatter(x = -235,y = 40)
        plt.xlim(-260,260)
        plt.ylim(-50, 425)
        st.pyplot(plt)
    
    ## MEDIA
  
   





    # Video with URL
    st.subheader("une vidéo directement de YouTube:")
    st.video(data='https://www.youtube.com/watch?v=85qE4J_jBMo', start_time = 295)

if choix_partie == 'VII - Conclusion et Perspectives':
    st.subheader('VII - Conclusion et Perspectives')
    
    st.metric(label = 'Meilleur Score du modèle',value = np.round(np.load('score_XGBoost.npy'),2))
    
    liste = ['Distribution','Shap_Value','%shot made flag']
    choix = st.radio('''Type d'analyse de la variable action_type''',liste)
    if choix == 'Distribution':
        plt.figure(figsize = (15,10))
        graph = df_viz['action_type'].value_counts(normalize=True).mul(100).reset_index()\
    .rename(columns = {'action_type' : 'percentage', 'index' : 'action_type',})
        p = sns.barplot(x="action_type", y="percentage",order = graph.sort_values('percentage', ascending = False)['action_type'], data = graph)
        graph = plt.setp(p.get_xticklabels(), rotation=90) 
        sns.set_style('white')
        plt.xlabel(None)
        plt.title('Distribution de la variable action_type');
        st.pyplot(plt)
        
    if choix == 'Shap_Value':
        col1,col2 = st.columns(2)
        with col1:
            st.image(Image.open('summary_plot_bar.png'))
            
        with col2:
            st.image(Image.open('summary_plot_dot.png'))
    
    if choix == '%shot made flag':
        plt.figure(figsize = (10,3))
        graph = df_viz[df_viz['action_type'] == 'Jump Shot']['shot_made_flag'].value_counts(normalize = True).reset_index().rename(columns={'shot_made_flag': '% de réussite au tir','index' : 'shot_made_flag'})
        graph['shot_made_flag'].replace([0,1],['tir manqué','tir réussi'], inplace = True)
        p = sns.barplot(data = graph, x = 'shot_made_flag',y = '% de réussite au tir')
        plt.bar_label(p.containers[0],fmt = '%.2f', label_type = 'center', fontsize = 15)
        st.pyplot(plt);
        
    
    st.subheader('''Perspectives d'améliorations:''')

    st.markdown(''':question: Données de défense liées au tir (distance du défenseur le plus proche, angle, la pression défensive exercée, quel défenseur et sa taille;''')
    st.markdown(''':question: Temps restant sur la possession;''')
    st.markdown(''':question: Historique de la réussite au tir selon chaque position sur le terrain;''')
    st.markdown(''':question: etc...''')
      
