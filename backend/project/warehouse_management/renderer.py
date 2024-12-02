import pygame
import numpy as np
from warehouse_management import env

# Dimensions de la grille et des cellules
GRID_SIZE = (10, 10)
CELL_SIZE = 50  # Taille en pixels d'une cellule
WINDOW_SIZE = (GRID_SIZE[1] * CELL_SIZE, GRID_SIZE[0] * CELL_SIZE)

# Couleurs et images associées aux types de cellules
COLORS = {
    1: (255, 255, 255),  # EMPTY: white
    0: (128, 128, 128),  # OBSTACLE: grey
    2: None,  # AGENT_WITHOUT_OBJECT: agent image
    3: None,  # AGENT_WITH_PRIMARY: agent with primary object image
    4: None,  # AGENT_WITH_SECONDARY: agent with secondary object image
    5: None,  # PRIMARY_OBJECT: primary object image
    6: None,  # SECONDARY_OBJECT: secondary object image
    7: (173, 216, 230),  # EMPTY_INPUT: light blue
    8: (173, 216, 230),  # INPUT_WITH_OBJECT: input object image
    9: (255, 200, 128),  # EMPTY_INPUT_CRAFT: light orange
    10: (255, 200, 128),  # INPUT_CRAFT_WITH_OBJECT: input craft image
    11: (255, 165, 0),  # EMPTY_OUTPUT_CRAFT: dark orange
    12: (255, 165, 0),  # OUTPUT_CRAFT_WITH_OBJECT: output craft image
    13: (255, 182, 193),  # EMPTY_OUTPUT: light red
    14: (255, 182, 193),  # OUTPUT_WITH_OBJECT: output object image
}

IMAGES = {
    2: "asset/agent.png",
    3: "asset/agent_primary_object.png",
    4: "asset/agent_secondary_object.png",
    5: "asset/primary_object.png",
    6: "asset/secondary_object.png",
    7: "asset/input.png",
    8: "asset/primary_object.png",
    9: "asset/input.png",
    10: "asset/primary_object.png",
    11: "asset/output.png",
    12: "asset/secondary_object.png",
    13: "asset/output.png",
    14: "asset/secondary_object.png",
}

# Mapping des touches pour les actions
KEY_ACTIONS = {
    pygame.K_UP: 1,     # Move up
    pygame.K_DOWN: 2,   # Move down
    pygame.K_LEFT: 3,   # Move left
    pygame.K_RIGHT: 4,  # Move right
    pygame.K_a: 5,  # Pick up
    pygame.K_q: 6  # Drop
}

def draw_grid(screen, grid, agent_positions):
    """
    Dessine la grille en fonction de l'état actuel.
    Args:
        screen (pygame.Surface): Surface de la fenêtre PyGame.
        grid (numpy.ndarray): La grille représentant l'environnement.
        agent_positions (dict): Positions des agents dans la grille.
    """
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell_value = grid[row, col]
            cell_rect = pygame.Rect(
                col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # Dessiner la couleur de la cellule
            color = COLORS.get(cell_value, (0, 0, 0))  # Noir par défaut
            if color:
                pygame.draw.rect(screen, color, cell_rect)

            # Dessiner une image si elle existe
            image_path = IMAGES.get(cell_value)
            if image_path:
                image = pygame.image.load(image_path)
                image = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
                screen.blit(image, (col * CELL_SIZE, row * CELL_SIZE))

    # Dessiner les agents par-dessus la grille
    for agent, (row, col) in agent_positions.items():
        agent_image_path = IMAGES.get(grid[row, col], "asset/agent.png")
        agent_image = pygame.image.load(agent_image_path)
        agent_image = pygame.transform.scale(
            agent_image, (CELL_SIZE, CELL_SIZE))
        screen.blit(agent_image, (col * CELL_SIZE, row * CELL_SIZE))


def run_visualization():
    """
    Lance l'environnement visuel avec PyGame.
    Permet de contrôler un agent manuellement via les flèches du clavier.
    """
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Warehouse Management Visualization")

    # Charger l'environnement
    warehouse_env = env(grid_size=GRID_SIZE, agents_number=3)
    observations = warehouse_env.reset()

    # Agent à contrôler manuellement
    manual_agent = "agent_0"
    clock = pygame.time.Clock()
    running = True

    while running:
        actions = {}

        # Gestion des événements (clavier)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Assigner une action à l'agent contrôlé manuellement
                if event.key in KEY_ACTIONS:
                    actions[manual_agent] = KEY_ACTIONS[event.key]

        # Actions automatiques pour les autres agents
        for agent in warehouse_env.agents:
            if agent != manual_agent:
                actions[agent] = warehouse_env.action_space[agent].sample()

        # Effectuer une étape dans l'environnement
        observations, rewards, terminated, truncated = warehouse_env.step(actions)

        if True in terminated.values():
            break

        # Effacer l'écran
        screen.fill((0, 0, 0))

        # Dessiner la grille
        draw_grid(screen, warehouse_env.grid, warehouse_env.agent_positions)

        # Afficher les scores en haut de l'écran
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Rewards: {rewards[manual_agent]}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        # Mettre à jour l'écran
        pygame.display.flip()

        # Contrôle de la vitesse de simulation
        clock.tick(5)  # 5 FPS

    warehouse_env.close()
    pygame.quit()


if __name__ == "__main__":
    run_visualization()
