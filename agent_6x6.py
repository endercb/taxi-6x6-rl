import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class EnhancedTaxi6x6(gym.Env):
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(self):
        self.grid_size = 6
        # Hedef lokasyonlar: R (0,0), G (0,5), Y (5,0), B (5,5), P (2,2)
        self.locs = [(0, 0), (0, 5), (5, 0), (5, 5), (2, 2)]
        self.dest_names = ['R', 'G', 'Y', 'B', 'P']
        self.dest_colors = ['red', 'green', 'orange', 'blue', 'purple']
        
        self.walls = {
            (0, 1): ['E'], (0, 2): ['W'],
            (1, 1): ['E'], (1, 2): ['W'],
            (2, 1): ['E'], (2, 2): ['W'],
            (4, 3): ['E'], (4, 4): ['W'],
            (5, 3): ['E'], (5, 4): ['W']
        }
        
        self.observation_space = spaces.Discrete(1080)
        self.action_space = spaces.Discrete(6)
        self.state = None
        
    def encode(self, r, c, p_idx, d_idx):
        return r * 180 + c * 30 + p_idx * 5 + d_idx
        
    def decode(self, i):
        d_idx = i % 5
        i //= 5
        p_idx = i % 6
        i //= 6
        c = i % 6
        r = i // 6
        return r, c, p_idx, d_idx
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random.integers(0, 6)
        c = self.np_random.integers(0, 6)
        p_idx = self.np_random.integers(0, 5)
        d_idx = self.np_random.integers(0, 5)
        while p_idx == d_idx:
            d_idx = self.np_random.integers(0, 5)
            
        self.state = self.encode(r, c, p_idx, d_idx)
        return self.state, {}
        
    def step(self, action):
        r, c, p_idx, d_idx = self.decode(self.state)
        reward = -1
        terminated = False
        
        if action == 0: # Güney (South)
            if r < 5 and 'S' not in self.walls.get((r, c), []): r += 1
        elif action == 1: # Kuzey (North)
            if r > 0 and 'N' not in self.walls.get((r, c), []): r -= 1
        elif action == 2: # Doğu (East)
            if c < 5 and 'E' not in self.walls.get((r, c), []): c += 1
        elif action == 3: # Batı (West)
            if c > 0 and 'W' not in self.walls.get((r, c), []): c -= 1
        elif action == 4: # Yolcu Al (Pickup)
            if p_idx < 5 and r == self.locs[p_idx][0] and c == self.locs[p_idx][1]:
                p_idx = 5
            else:
                reward = -10
        elif action == 5: # Yolcu Bırak (Dropoff)
            if p_idx == 5 and r == self.locs[d_idx][0] and c == self.locs[d_idx][1]:
                p_idx = d_idx
                reward = 20
                terminated = True
            elif p_idx == 5 and (r, c) in self.locs:
                p_idx = self.locs.index((r, c))
            else:
                reward = -10
                
        self.state = self.encode(r, c, p_idx, d_idx)
        return self.state, reward, terminated, False, {}

    def render(self):
        r, c, p_idx, d_idx = self.decode(self.state)
        cell_size = 60
        img_size = self.grid_size * cell_size
        img = Image.new('RGB', (img_size, img_size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Izgarayı çiz
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0, y0 = j * cell_size, i * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                draw.rectangle([x0, y0, x1, y1], outline='lightgray')
                
        # Duvarları çiz
        for (wr, wc), w_dirs in self.walls.items():
            x, y = wc * cell_size, wr * cell_size
            if 'E' in w_dirs: draw.line([x+cell_size, y, x+cell_size, y+cell_size], fill='black', width=4)
            if 'W' in w_dirs: draw.line([x, y, x, y+cell_size], fill='black', width=4)
            if 'N' in w_dirs: draw.line([x, y, x+cell_size, y], fill='black', width=4)
            if 'S' in w_dirs: draw.line([x, y+cell_size, x+cell_size, y+cell_size], fill='black', width=4)

        # Hedefleri çiz (Kare şeklinde)
        for i, (lr, lc) in enumerate(self.locs):
            x, y = lc * cell_size, lr * cell_size
            draw.rectangle([x+5, y+5, x+55, y+55], outline=self.dest_colors[i], width=3)
            draw.text((x + 25, y + 25), self.dest_names[i], fill=self.dest_colors[i])
            
        # Hedef olan noktayı vurgula (Daha kalın kenarlık)
        if p_idx < 5: 
            target_r, target_c = self.locs[d_idx]
            tx, ty = target_c * cell_size, target_r * cell_size
            draw.rectangle([tx+2, ty+2, tx+58, ty+58], outline='red', width=5)

        # Taksi Çizimi
        tx, ty = c * cell_size, r * cell_size
        taxi_color = 'gold' if p_idx < 5 else 'limegreen' # Yolcu taksideyse yeşil, değilse sarı
        draw.rectangle([tx+15, ty+15, tx+45, ty+45], fill=taxi_color, outline='black', width=2)
        
        # Yolcu Bekliyorsa Çiz (Taksi içinde değilse)
        if p_idx < 5:
            px, py = self.locs[p_idx][1] * cell_size, self.locs[p_idx][0] * cell_size
            draw.ellipse([px+20, py+20, px+40, py+40], fill='blue')
        
        return img

def train_agent():
    env = EnhancedTaxi6x6()
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    epochs = 20000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.01
    
    rewards_all_episodes = []
    
    print("Eğitim Başlıyor (20.000 Epoch)...")
    for epoch in range(epochs):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Q-Learning Update
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            
            state = next_state
            total_reward += reward
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_all_episodes.append(total_reward)
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch: {epoch + 1}/{epochs} | Son 100 Epoch Ort. Ödül: {np.mean(rewards_all_episodes[-100:]):.2f}")
            
    print("Eğitim Tamamlandı!")
    
    # Eğitim Grafiğini Çiz
    smoothed_rewards = [np.mean(rewards_all_episodes[i:i+100]) for i in range(0, len(rewards_all_episodes), 100)]
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(rewards_all_episodes), 100), smoothed_rewards, color='blue', linewidth=2)
    plt.title('Training Rewards over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig('training_graph.png')
    plt.close()
    print("Eğitim grafiği 'training_graph.png' olarak kaydedildi.")
    
    # Test ve GIF Oluşturma
    print("Test Senaryosu Başlatılıyor ve GIF Oluşturuluyor...")
    
    def run_test_scenario(start_r, start_c, pass_idx, dest_idx, filename):
        state = env.encode(start_r, start_c, pass_idx, dest_idx)
        env.state = state
        frames = []
        frames.append(env.render())
        done = False
        steps = 0
        total_test_reward = 0
        while not done and steps < 50:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_test_reward += reward
            steps += 1
            frames.append(env.render())
        
        if len(frames) > 0:
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=300,
                loop=0
            )
            print(f"GIF '{filename}' olarak kaydedildi! (Adım: {steps}, Ödül: {total_test_reward})")

    # Senaryo 1: Rastgele boş hücreden başlasın
    run_test_scenario(3, 4, 0, 2, "taxi_test_1.gif")
    
    # Senaryo 2: Başka bir boş hücreden başlasın, farklı bir yolcuyu alsın
    run_test_scenario(1, 4, 3, 1, "taxi_test_2.gif")

if __name__ == '__main__':
    train_agent()