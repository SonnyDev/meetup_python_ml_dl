import streamlit as st
import numpy as np
import pandas as pd
import time
import random

# Configuration de la page Streamlit
st.set_page_config(page_title="Q-Learning Visualizer", layout="wide")
st.title("Visualisation Interactive du Q-Learning")

# Sidebar pour les hyperparamètres
with st.sidebar:
    st.header("Paramètres")
    ALPHA = st.slider("Taux d'apprentissage (α)", 0.0, 1.0, 0.5)
    EPSILON = st.slider("Taux d'exploration (ε)", 0.0, 1.0, 0.9)
    GAMMA = st.slider("Facteur d'actualisation (γ)", 0.0, 1.0, 0.9)
    
    if st.button("Réinitialiser l'apprentissage"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Paramètres de l'environnement
STATES = list(range(1, 10))  # États de 1 à 9
ACTIONS = ["Haut", "Bas", "Gauche", "Droite"]
GOAL_STATE = 9
HOLE_STATE = 5
INITIAL_STATE = 1

# Ajout d'une variable pour stocker la dernière action
if "last_action_taken" not in st.session_state:
    st.session_state.last_action_taken = None

# Fonction pour obtenir l'état suivant et la récompense
def get_next_state_reward(state, action):
    # Matrice de transition
    transitions = {
        1: {"Haut": 1, "Bas": 4, "Gauche": 1, "Droite": 2},
        2: {"Haut": 2, "Bas": 5, "Gauche": 1, "Droite": 3},
        3: {"Haut": 3, "Bas": 6, "Gauche": 2, "Droite": 3},
        4: {"Haut": 1, "Bas": 7, "Gauche": 4, "Droite": 5},
        5: {"Haut": 2, "Bas": 8, "Gauche": 4, "Droite": 6},  # Trou
        6: {"Haut": 3, "Bas": 9, "Gauche": 5, "Droite": 6},
        7: {"Haut": 4, "Bas": 7, "Gauche": 7, "Droite": 8},
        8: {"Haut": 5, "Bas": 8, "Gauche": 7, "Droite": 9},
        9: {"Haut": 6, "Bas": 9, "Gauche": 8, "Droite": 9}  # But
    }
    
    next_state = transitions[state][action]
    
    if next_state == GOAL_STATE:
        return next_state, 1  # Récompense positive pour atteindre l'objectif
    elif next_state == HOLE_STATE:
        return next_state, -1  # Pénalité pour tomber dans le trou
    else:
        return next_state, 0  # Pas de récompense pour les autres transitions

# Initialisation de la Q-table
def initialize_q_table():
    # Initialisation optimiste avec des valeurs autour de 20
    base_value = 20
    q_table = {}
    for state in STATES:
        if state == GOAL_STATE:
            # État terminal (but) initialisé à 0
            q_table[state] = {
                "Haut": 0,
                "Bas": 0,
                "Gauche": 0,
                "Droite": 0
            }
        elif state == HOLE_STATE:
            # État du trou initialisé avec des valeurs négatives
            q_table[state] = {
                "Haut": -base_value,
                "Bas": -base_value,
                "Gauche": -base_value,
                "Droite": -base_value
            }
        else:
            # Autres états initialisés avec des valeurs optimistes
            q_table[state] = {
                "Haut": base_value + random.uniform(-2, 2),
                "Bas": base_value + random.uniform(-2, 2),
                "Gauche": base_value + random.uniform(-2, 2),
                "Droite": base_value + random.uniform(-2, 2)
            }
    return q_table

# Affichage de la Q-table sous forme de grille colorée
def display_q_table(q_table, current_state=None):
    df = pd.DataFrame.from_dict(q_table, orient='index')
    
    # Formatage des valeurs pour n'afficher que 2 décimales
    df = df.round(2)
    
    # Style de base avec le gradient de couleurs
    styler = df.style.background_gradient(
        cmap='RdBu',
        vmin=df.values.mean() - 2 * df.values.std(),
        vmax=df.values.max() + 2 * df.values.std(),
        axis=None
    )
    
    # Si un état courant est spécifié
    if current_state is not None:
        # Mettre en évidence la ligne de l'état courant (sans soustraire 1)
        styler.set_properties(
            subset=pd.IndexSlice[current_state:current_state, :],
            **{'background-color': 'yellow', 'font-weight': 'bold'}
        )
        
        # Mettre en évidence la meilleure action pour l'état courant
        best_action = max(q_table[current_state].items(), key=lambda x: x[1])[0]
        styler.set_properties(
            subset=pd.IndexSlice[current_state:current_state, best_action],
            **{'color': 'green', 'font-weight': 'bold', 'border': '2px solid green'}
        )
    
    st.dataframe(
        styler,
        use_container_width=True,
        height=400
    )

# Affichage de la grille de jeu sous forme de tableau
def display_grid(current_state, q_table=None):
    grid = []
    for i in range(3):
        row = []
        for j in range(3):
            state = i * 3 + j + 1
            
            if state == current_state:
                cell = "🤖"
                if st.session_state.last_action_taken:
                    arrows = {"Haut": "⬆️", "Bas": "⬇️", "Gauche": "⬅️", "Droite": "➡️"}
                    cell = f"{cell}{arrows[st.session_state.last_action_taken]}"
            elif state == GOAL_STATE:
                cell = "🎯"
            elif state == HOLE_STATE and current_state != HOLE_STATE:  # Ne montre le trou que si l'agent n'y est pas
                cell = "🕳️"
            else:
                cell = "⬜"
            row.append(cell)
        grid.append(row)
    
    df = pd.DataFrame(grid)
    st.dataframe(df, use_container_width=True, height=150)

def run_multiple_iterations(n_iterations, q_table, initial_state):
    state = initial_state
    for _ in range(n_iterations):
        # Politique epsilon-greedy
        if random.random() >= EPSILON:
            action = max(q_table[state].items(), key=lambda x: x[1])[0]
        else:
            action = random.choice(ACTIONS)
        
        # Obtention du nouvel état et de la récompense
        next_state, reward = get_next_state_reward(state, action)
        
        # Mise à jour Q-table
        old_value = q_table[state][action]
        next_max = max(q_table[next_state].values())
        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[state][action] = new_value
        
        # Réinitialisation si état terminal
        if next_state in [GOAL_STATE, HOLE_STATE]:
            state = INITIAL_STATE
        else:
            state = next_state
    
    return q_table

# Algorithme Q-Learning avec bouton pour avancer étape par étape
def q_learning():
    # Initialisation du session_state
    if "q_table" not in st.session_state:
        st.session_state.q_table = initialize_q_table()
        st.session_state.state = INITIAL_STATE
        st.session_state.steps = 0
        st.session_state.total_reward = 0
        st.session_state.episode = 1
        st.session_state.episode_rewards = []
        st.session_state.done = False

    # Récupération des informations
    q_table = st.session_state.q_table
    state = st.session_state.state
    steps = st.session_state.steps
    total_reward = st.session_state.total_reward

    # Interface
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### Environnement")
        display_grid(state, q_table)
        
        # Affichage des métriques avec labels visibles
        m1, m2, m3 = st.columns(3)
        m1.metric("Episode", st.session_state.episode)
        m2.metric("Reward", round(total_reward, 2))
        m3.metric("Steps", steps)
    
    with col2:
        st.markdown("### Q-Table")
        display_q_table(q_table, state)
    
    with col3:
        st.markdown("### Contrôles")
        
        # Mode pas à pas
        st.markdown("#### Pas à pas")
        if st.button("Étape suivante", key="step", use_container_width=True):
            # Trouver la meilleure action pour l'état actuel
            best_action = max(q_table[state].items(), key=lambda x: x[1])[0]
            
            # Politique epsilon-greedy avec l'epsilon de la sidebar
            if random.random() >= EPSILON:  # Exploitation
                action = best_action  # Utiliser la meilleure action
            else:  # Exploration
                action = random.choice(ACTIONS)  # Action complètement aléatoire
            
            # Stockage de l'action prise
            st.session_state.last_action_taken = action
            
            # Afficher le mode et les valeurs Q
            if action == best_action:
                st.info(f"Mode: Exploitation - Action: {action} (Q={q_table[state][action]:.2f})")
            else:
                st.info(f"Mode: Exploration - Action: {action} (Q={q_table[state][action]:.2f})")
            
            # Obtention du nouvel état et de la récompense
            next_state, reward = get_next_state_reward(state, action)
            total_reward += reward
            
            # Mise à jour de la Q-table
            old_value = q_table[state][action]
            next_max = max(q_table[next_state].values())
            new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
            q_table[state][action] = new_value
            
            # Mise à jour du session_state
            st.session_state.q_table = q_table
            st.session_state.state = next_state
            st.session_state.steps = steps + 1
            st.session_state.total_reward = total_reward
            
            # Gestion des fins d'épisode
            if next_state == GOAL_STATE:
                st.session_state.last_action = "goal"
                st.rerun()  # Montrer d'abord l'agent sur l'objectif
            elif next_state == HOLE_STATE:
                st.session_state.last_action = "hole"
                st.rerun()  # Montrer d'abord l'agent dans le trou
            
            st.rerun()
        
        # Mode exécution rapide
        st.markdown("#### Rapide")
        n_iterations = st.number_input("Nombre d'itérations", min_value=1, value=1000)
        if st.button("Exécuter", key="run_fast", use_container_width=True):
            # Exécuter les itérations
            final_q_table = run_multiple_iterations(
                n_iterations, 
                st.session_state.q_table.copy(), 
                INITIAL_STATE
            )
            
            # Mettre à jour la Q-table
            st.session_state.q_table = final_q_table
            
            # Afficher les statistiques de la Q-table finale
            avg_value = np.mean([max(values.values()) for values in final_q_table.values()])
            max_value = np.max([max(values.values()) for values in final_q_table.values()])
            min_value = np.min([min(values.values()) for values in final_q_table.values()])
            
            st.write("Statistiques de la Q-table :")
            col1, col2, col3 = st.columns(3)
            col1.metric("Valeur moyenne", f"{avg_value:.2f}")
            col2.metric("Valeur max", f"{max_value:.2f}")
            col3.metric("Valeur min", f"{min_value:.2f}")
            
            st.rerun()
        
        # Bouton de réinitialisation (remonté, juste un petit espace)
        st.write("")  # Un seul espace au lieu du divider
        if st.button("Réinitialiser", key="reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.last_action_taken = None
            st.rerun()
        
        # Messages de statut
        if "last_action" in st.session_state:
            if st.session_state.last_action == "goal":
                st.success("🎯 But atteint!", icon="✅")
                # Réinitialisation après avoir montré le succès
                st.session_state.state = INITIAL_STATE
                st.session_state.last_action_taken = None
                st.session_state.last_action = None
            elif st.session_state.last_action == "hole":
                st.error("🕳️ Tombé dans le trou!", icon="🚫")
                # Réinitialisation après avoir montré l'échec
                st.session_state.state = INITIAL_STATE
                st.session_state.last_action_taken = None
                st.session_state.last_action = None

# Démarrage automatique
if "q_table" not in st.session_state:
    q_learning()
else:
    q_learning()
