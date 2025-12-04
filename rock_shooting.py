import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import random

# --- 1. 參數設定 ---
WIDTH = 800
HEIGHT = 600
FPS = 60
GAME_DURATION = 80

# 投擲冷卻時間 (幀數)
THROW_COOLDOWN_FRAMES = 15 

# 顏色定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
YELLOW = (255, 215, 0)
GREY = (150, 150, 150)
BLUE = (100, 200, 255)
PURPLE = (200, 50, 255) 
CYAN = (0, 255, 255) 
ORANGE = (255, 165, 0) 
GOLD = (255, 223, 0) 
DARK_GREEN = (0, 100, 0)
DARK_ORANGE = (200, 100, 0)

# 遊戲參數
SENSITIVITY = 1.8      
BLINK_THRESHOLD = 0.22 
PINCH_THRESHOLD = 0.035 

# 初始化
pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()
pygame.mixer.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("雙模組投擲 - 極速優化版")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24, bold=True)
big_font = pygame.font.SysFont("Arial", 48, bold=True)
game_over_font = pygame.font.SysFont("Arial", 64, bold=True)
bonus_font = pygame.font.SysFont("Arial", 32, bold=True)

# --- MediaPipe 模型初始化 (效能優化) ---

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0, # 【優化重點 1】 使用輕量級模型 (0=Lite, 1=Full)
    min_detection_confidence=0.5, # 稍微降低門檻以換取速度
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# --- 2. 音效生成 ---

def generate_throw_sound():
    duration = 0.3
    sample_rate = 44100
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    freqs = np.linspace(300, 800, n_samples)
    waveform = np.sin(2 * np.pi * freqs * t)
    envelope = np.exp(-3 * t) 
    waveform *= envelope * 0.5
    sound_data = (waveform * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_data)

def generate_hit_sound():
    duration = 0.15
    sample_rate = 44100
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    noise = np.random.uniform(-1, 1, n_samples)
    envelope = np.exp(-20 * t)
    waveform = noise * envelope * 0.6
    sound_data = (waveform * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_data)

def generate_combo_sound():
    duration = 0.2
    sample_rate = 44100
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    freqs = np.linspace(800, 1200, n_samples) 
    waveform = np.sin(2 * np.pi * freqs * t)
    envelope = np.exp(-10 * t)
    waveform *= envelope * 0.4
    sound_data = (waveform * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_data)

def generate_select_sound():
    duration = 0.4
    sample_rate = 44100
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    freqs = np.linspace(400, 600, n_samples)
    waveform = np.sin(2 * np.pi * freqs * t)
    envelope = np.ones_like(t)
    waveform *= envelope * 0.5
    sound_data = (waveform * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_data)

throw_fx = generate_throw_sound()
hit_fx = generate_hit_sound()
combo_fx = generate_combo_sound()
select_fx = generate_select_sound()

# --- 3. 類別設計 ---

def calculate_ear(landmarks, indices):
    v1 = np.linalg.norm(np.array(landmarks[indices[1]]) - np.array(landmarks[indices[5]]))
    v2 = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[4]]))
    h = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[3]]))
    ear = (v1 + v2) / (2.0 * h)
    return ear

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.radius = random.randint(3, 8)
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.gravity = 0.2
        self.life = 40 
        self.alpha = 255 

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity 
        self.life -= 1
        self.alpha = max(0, self.alpha - 6) 

    def draw(self, surface):
        if self.life > 0:
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, self.alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (self.x - self.radius, self.y - self.radius))

class FloatingText:
    def __init__(self, x, y, text, color):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.life = 40
        self.vy = -2

    def update(self):
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            txt_surf = bonus_font.render(self.text, True, self.color)
            if self.life < 15:
                alpha = int((self.life / 15) * 255)
                txt_surf.set_alpha(alpha)
            surface.blit(txt_surf, (self.x, self.y))

class Target:
    def __init__(self, is_special=False):
        self.is_special = is_special
        if self.is_special:
            self.type = 'special'
            self.radius = 60 
            self.color = PURPLE
            self.points = 50 
            self.center_x = WIDTH // 2
            self.center_y = HEIGHT // 2 - 50
            self.orbit_radius = 200 
            self.angle = 0
            self.angular_speed = 0.024 
            self.x = self.center_x + math.cos(self.angle) * self.orbit_radius
            self.y = self.center_y + math.sin(self.angle) * self.orbit_radius
            self.life_timer = 300 
        else:
            self.type = random.choice(['fixed', 'moving'])
            if self.type == 'fixed':
                self.radius = random.randint(30, 50)
                self.color = RED
                self.points = 10
                self.x = random.randint(self.radius, WIDTH - self.radius)
                self.y = random.randint(self.radius, HEIGHT - 200)
                self.vx = 0
            else:
                self.radius = random.randint(25, 45)
                self.color = YELLOW
                self.points = 20
                self.y = random.randint(50, HEIGHT - 250)
                if random.random() < 0.5:
                    self.x = -self.radius
                    self.vx = random.randint(2, 5)
                else:
                    self.x = WIDTH + self.radius
                    self.vx = random.randint(-5, -2)
        self.active = True

    def update(self):
        if self.is_special:
            self.angle += self.angular_speed
            self.x = self.center_x + math.cos(self.angle) * self.orbit_radius
            self.y = self.center_y + math.sin(self.angle) * self.orbit_radius
            self.life_timer -= 1
            if self.life_timer <= 0:
                self.active = False
        else:
            self.x += self.vx
            if self.type == 'moving':
                if (self.vx > 0 and self.x > WIDTH + 100) or (self.vx < 0 and self.x < -100):
                    self.active = False

    def draw(self, surface):
        if self.active:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), int(self.radius * 0.7))
            if self.is_special:
                 pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y)), int(self.radius * 0.9), 3)
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius * 0.4))

class Stone:
    def __init__(self, target_x, target_y):
        self.start_x = WIDTH // 2
        self.start_y = HEIGHT
        self.end_x = target_x
        self.end_y = target_y
        self.x = self.start_x
        self.y = self.start_y
        self.progress = 0.0 
        self.speed = 0.035 
        self.arc_height = 200 
        self.start_radius = 30
        self.end_radius = 8
        self.current_radius = self.start_radius
        self.active = True
        self.hit = False

    def update(self):
        self.progress += self.speed
        if self.progress >= 1.0:
            self.active = False
            return 

        linear_x = self.start_x + (self.end_x - self.start_x) * self.progress
        linear_y = self.start_y + (self.end_y - self.start_y) * self.progress
        arc_offset = math.sin(self.progress * math.pi) * self.arc_height
        
        self.x = linear_x
        self.y = linear_y - arc_offset
        self.current_radius = self.start_radius * (1 - self.progress) + self.end_radius * self.progress

    def draw(self, surface):
        if self.active:
            pygame.draw.circle(surface, (50, 50, 50), (int(self.x + 2), int(self.y + 2)), int(self.current_radius))
            pygame.draw.circle(surface, GREY, (int(self.x), int(self.y)), int(self.current_radius))

# --- 4. 主程式 ---

def main():
    running = True
    
    cursor_x, cursor_y = WIDTH // 2, HEIGHT // 2
    fire_cooldown = 0
    is_firing_gesture = False 
    
    # 手部模式狀態
    pinch_states = {} 
    
    game_state = "menu" 
    
    score = 0
    total_throws = 0
    total_hits = 0
    consecutive_hits = 0
    
    stones = []
    targets = []
    particles = []
    floating_texts = [] 
    
    start_ticks = 0 
    spawn_timer = 0
    next_special_spawn_time = 0
    TARGET_LIMIT = 5
    
    post_recording_frames = 0 
    
    control_mode = "EYE" 
    
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    CENTER_POINT = 168

    video_writer = None
    RECORD_FILENAME = "spinning_recording.mp4"

    btn_width, btn_height = 250, 200
    eye_btn_rect = pygame.Rect(WIDTH//2 - btn_width - 20, HEIGHT//2 - 50, btn_width, btn_height)
    hand_btn_rect = pygame.Rect(WIDTH//2 + 20, HEIGHT//2 - 50, btn_width, btn_height)

    def init_menu():
        nonlocal game_state, cursor_x, cursor_y, video_writer
        game_state = "menu"
        cursor_x, cursor_y = WIDTH // 2, HEIGHT // 2
        if video_writer is not None:
            video_writer.release()
            video_writer = None

    def start_game(mode):
        nonlocal score, stones, targets, particles, floating_texts, start_ticks, game_state, next_special_spawn_time, total_throws, total_hits, consecutive_hits, video_writer, control_mode, post_recording_frames, pinch_states
        
        control_mode = mode
        score = 0
        total_throws = 0
        total_hits = 0
        consecutive_hits = 0
        stones.clear()
        targets.clear()
        particles.clear()
        floating_texts.clear()
        pinch_states = {} 
        game_state = "playing"
        start_ticks = pygame.time.get_ticks()
        next_special_spawn_time = GAME_DURATION - 15
        post_recording_frames = 120

        if video_writer is not None:
            video_writer.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(RECORD_FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
        print(f"Game Started ({mode}). Recording started.")
        select_fx.play()

    init_menu()

    while running:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w, _ = frame.shape
        
        # 繪圖背景
        frame_bg = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_bg_rgb = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(frame_bg_rgb)))
        screen.blit(frame_surface, (0, 0))
        
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(100)
        overlay.fill(BLACK)
        screen.blit(overlay, (0,0))

        # 【優化重點 2】 變數初始化，不預先執行 Face Mesh
        results_face = None
        results_hands = None

        # -----------------------------------------------------
        # 狀態：選單 (MENU) - 必須使用 Face Mesh
        # -----------------------------------------------------
        if game_state == "menu":
            # 選單模式：執行臉部偵測
            results_face = face_mesh.process(rgb_frame)
            
            title_text = game_over_font.render("SELECT MODE", True, WHITE)
            subtitle_text = font.render("Use HEAD to aim, BLINK to select", True, CYAN)
            screen.blit(title_text, (WIDTH//2 - 200, 50))
            screen.blit(subtitle_text, (WIDTH//2 - 180, 120))

            if results_face.multi_face_landmarks:
                landmarks = results_face.multi_face_landmarks[0].landmark
                center_lm = landmarks[CENTER_POINT]
                cx, cy = int(center_lm.x * frame_w), int(center_lm.y * frame_h)
                offset_x = (cx - frame_w / 2) * SENSITIVITY
                offset_y = (cy - frame_h / 2) * SENSITIVITY
                target_cursor_x = (WIDTH / 2) + offset_x * (WIDTH / frame_w * 4)
                target_cursor_y = (HEIGHT / 2) + offset_y * (HEIGHT / frame_h * 4)
                cursor_x += (target_cursor_x - cursor_x) * 0.3 
                cursor_y += (target_cursor_y - cursor_y) * 0.3
                
                mesh_points = np.array([[p.x, p.y] for p in landmarks])
                left_ear = calculate_ear(mesh_points, LEFT_EYE)
                right_ear = calculate_ear(mesh_points, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                is_blinking_now = (avg_ear < BLINK_THRESHOLD)
            else:
                is_blinking_now = False

            cursor_x = max(0, min(WIDTH, cursor_x))
            cursor_y = max(0, min(HEIGHT, cursor_y))

            btn1_color = DARK_GREEN
            if eye_btn_rect.collidepoint(cursor_x, cursor_y):
                btn1_color = GREEN 
                if is_blinking_now and fire_cooldown == 0:
                    start_game("EYE")
            
            pygame.draw.rect(screen, btn1_color, eye_btn_rect, border_radius=15)
            pygame.draw.rect(screen, WHITE, eye_btn_rect, 3, border_radius=15)
            txt_eye_1 = big_font.render("EYE", True, WHITE)
            txt_eye_2 = font.render("Blink to Throw", True, WHITE)
            screen.blit(txt_eye_1, (eye_btn_rect.centerx - 40, eye_btn_rect.centery - 20))
            screen.blit(txt_eye_2, (eye_btn_rect.centerx - 70, eye_btn_rect.centery + 40))

            btn2_color = DARK_ORANGE
            if hand_btn_rect.collidepoint(cursor_x, cursor_y):
                btn2_color = ORANGE 
                if is_blinking_now and fire_cooldown == 0:
                    start_game("HAND")

            pygame.draw.rect(screen, btn2_color, hand_btn_rect, border_radius=15)
            pygame.draw.rect(screen, WHITE, hand_btn_rect, 3, border_radius=15)
            txt_hand_1 = big_font.render("HAND", True, WHITE)
            txt_hand_2 = font.render("Pinch to Throw", True, WHITE)
            screen.blit(txt_hand_1, (hand_btn_rect.centerx - 60, hand_btn_rect.centery - 20))
            screen.blit(txt_hand_2, (hand_btn_rect.centerx - 75, hand_btn_rect.centery + 40))

            pygame.draw.circle(screen, GREEN, (int(cursor_x), int(cursor_y)), 15, 2)
            pygame.draw.circle(screen, RED, (int(cursor_x), int(cursor_y)), 3)
            
            if fire_cooldown > 0: fire_cooldown -= 1

        # -----------------------------------------------------
        # 狀態：遊戲中 (PLAYING)
        # -----------------------------------------------------
        elif game_state == "playing":
            seconds_passed = (pygame.time.get_ticks() - start_ticks) / 1000
            time_left = GAME_DURATION - seconds_passed
            
            if time_left <= 0:
                time_left = 0
                game_state = "game_over"

            active_hand_cursors = [] 
            trigger_throw = False
            thumb_pos = None
            index_pos = None

            # 【優化重點 3】 分流執行，只跑需要的模型
            if control_mode == "EYE":
                # 只執行 Face Mesh
                results_face = face_mesh.process(rgb_frame)
                
                if results_face.multi_face_landmarks:
                    landmarks = results_face.multi_face_landmarks[0].landmark
                    center_lm = landmarks[CENTER_POINT]
                    cx, cy = int(center_lm.x * frame_w), int(center_lm.y * frame_h)
                    offset_x = (cx - frame_w / 2) * SENSITIVITY
                    offset_y = (cy - frame_h / 2) * SENSITIVITY
                    target_cursor_x = (WIDTH / 2) + offset_x * (WIDTH / frame_w * 4)
                    target_cursor_y = (HEIGHT / 2) + offset_y * (HEIGHT / frame_h * 4)
                    cursor_x += (target_cursor_x - cursor_x) * 0.3 
                    cursor_y += (target_cursor_y - cursor_y) * 0.3
                    mesh_points = np.array([[p.x, p.y] for p in landmarks])
                    left_ear = calculate_ear(mesh_points, LEFT_EYE)
                    right_ear = calculate_ear(mesh_points, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0
                    if avg_ear < BLINK_THRESHOLD:
                        if not is_firing_gesture:
                            trigger_throw = True
                            is_firing_gesture = True
                    else:
                        is_firing_gesture = False

            elif control_mode == "HAND":
                # 只執行 Hands (完全跳過 Face Mesh)
                results_hands = hands.process(rgb_frame)
                
                if results_hands.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        hand_label = results_hands.multi_handedness[idx].classification[0].label
                        thumb_lm = hand_landmarks.landmark[4]
                        index_lm = hand_landmarks.landmark[8]
                        tx, ty = int(thumb_lm.x * WIDTH), int(thumb_lm.y * HEIGHT)
                        ix, iy = int(index_lm.x * WIDTH), int(index_lm.y * HEIGHT)
                        mid_x, mid_y = (tx + ix) / 2, (ty + iy) / 2
                        mid_x = max(0, min(WIDTH, mid_x))
                        mid_y = max(0, min(HEIGHT, mid_y))
                        active_hand_cursors.append(((mid_x, mid_y), (tx, ty), (ix, iy)))
                        
                        if time_left > 0:
                            dist_norm = math.hypot(thumb_lm.x - index_lm.x, thumb_lm.y - index_lm.y)
                            is_hand_pinching = pinch_states.get(hand_label, False)
                            
                            if dist_norm < PINCH_THRESHOLD:
                                if not is_hand_pinching:
                                    # 這裡也要判斷全域冷卻
                                    if fire_cooldown == 0:
                                        stones.append(Stone(mid_x, mid_y))
                                        total_throws += 1
                                        throw_fx.play()
                                        fire_cooldown = THROW_COOLDOWN_FRAMES
                                    pinch_states[hand_label] = True 
                            else:
                                pinch_states[hand_label] = False 

            cursor_x = max(0, min(WIDTH, cursor_x))
            cursor_y = max(0, min(HEIGHT, cursor_y))

            # 眼動模式的投擲邏輯
            if time_left > 0 and control_mode == "EYE":
                if trigger_throw and fire_cooldown == 0:
                    stones.append(Stone(cursor_x, cursor_y))
                    total_throws += 1
                    throw_fx.play()
                    fire_cooldown = THROW_COOLDOWN_FRAMES 

            if fire_cooldown > 0: fire_cooldown -= 1

            if time_left > 0:
                spawn_timer += 1
                if spawn_timer > 40:
                    if len([t for t in targets if not t.is_special]) < TARGET_LIMIT:
                        targets.append(Target(is_special=False))
                    spawn_timer = 0
                
                if time_left <= next_special_spawn_time and next_special_spawn_time > 0:
                    targets.append(Target(is_special=True))
                    next_special_spawn_time -= 15

            for t in targets[:]:
                t.update()
                if not t.active: targets.remove(t)
                    
            for s in stones[:]:
                s.update()
                if not s.active:
                    if not s.hit: consecutive_hits = 0 
                    stones.remove(s)
                    continue
                if s.progress > 0.9 and not s.hit:
                    hit_target = None
                    for t in targets:
                        distance = math.sqrt((s.x - t.x)**2 + (s.y - t.y)**2)
                        if distance < t.radius + s.current_radius:
                            hit_target = t
                            break
                    if hit_target:
                        s.hit = True
                        s.active = False
                        hit_target.active = False
                        consecutive_hits += 1
                        total_hits += 1
                        base_score = hit_target.points
                        bonus_score = 0
                        if consecutive_hits >= 2:
                            bonus_score = min(consecutive_hits, 5)
                            floating_texts.append(FloatingText(hit_target.x, hit_target.y - 40, f"Bonus +{bonus_score}", GOLD))
                            combo_fx.play()
                        else:
                            hit_fx.play()
                        score += base_score + bonus_score
                        for _ in range(15):
                            particles.append(Particle(hit_target.x, hit_target.y, hit_target.color))
                        stones.remove(s)
                        if hit_target in targets: targets.remove(hit_target)

            for p in particles[:]:
                p.update()
                if p.life <= 0: particles.remove(p)
            for ft in floating_texts[:]:
                ft.update()
                if ft.life <= 0: floating_texts.remove(ft)

            for t in targets: t.draw(screen)
            for p in particles: p.draw(screen)
            for s in stones: s.draw(screen)
            for ft in floating_texts: ft.draw(screen)
            
            if control_mode == "EYE":
                cx, cy = int(cursor_x), int(cursor_y)
                pygame.draw.circle(screen, GREEN, (cx, cy), 20, 2)
                pygame.draw.line(screen, GREEN, (cx - 30, cy), (cx + 30, cy), 2)
                pygame.draw.line(screen, GREEN, (cx, cy - 30), (cx, cy + 30), 2)
                pygame.draw.circle(screen, RED, (cx, cy), 4)
            else:
                for (mx, my), (tx, ty), (ix, iy) in active_hand_cursors:
                    pygame.draw.circle(screen, CYAN, (tx, ty), 5)
                    pygame.draw.circle(screen, CYAN, (ix, iy), 5)
                    pygame.draw.line(screen, GREY, (tx, ty), (ix, iy), 2)
                    pygame.draw.circle(screen, ORANGE, (int(mx), int(my)), 20, 2)
                    pygame.draw.line(screen, ORANGE, (int(mx) - 30, int(my)), (int(mx) + 30, int(my)), 2)
                    pygame.draw.line(screen, ORANGE, (int(mx), int(my) - 30), (int(mx), int(my) + 30), 2)
                    pygame.draw.circle(screen, WHITE, (int(mx), int(my)), 4)

            if total_throws > 0:
                accuracy = (total_hits / total_throws) * 100
            else:
                accuracy = 0.0

            score_text = big_font.render(f"Score: {score}", True, WHITE)
            screen.blit(score_text, (20, 20))
            if consecutive_hits >= 2:
                combo_text = big_font.render(f"Combo: {consecutive_hits}", True, GOLD)
                screen.blit(combo_text, (20, 140))
            acc_text = font.render(f"Accuracy: {accuracy:.1f}%", True, CYAN)
            screen.blit(acc_text, (20, 80))
            time_color = RED if time_left < 10 else BLUE
            time_text = big_font.render(f"Time: {int(time_left)}", True, time_color)
            screen.blit(time_text, (WIDTH - 220, 20))
            
            if video_writer is not None:
                if (pygame.time.get_ticks() // 500) % 2 == 0:
                    pygame.draw.circle(screen, RED, (WIDTH - 30, 30), 10)
                rec_text = font.render("REC", True, RED)
                screen.blit(rec_text, (WIDTH - 80, 20))

            mode_str = f"MODE: {control_mode}"
            mode_text = font.render(mode_str, True, WHITE)
            switch_text = font.render("[Press 'M' to Switch Mode]", True, GREY)
            screen.blit(mode_text, (WIDTH//2 - 50, HEIGHT - 60))
            screen.blit(switch_text, (WIDTH//2 - 110, HEIGHT - 30))

        elif game_state == "game_over":
            if total_throws > 0:
                accuracy = (total_hits / total_throws) * 100
            else:
                accuracy = 0.0

            go_text = game_over_font.render("TIME UP!", True, YELLOW)
            final_score = big_font.render(f"Score: {score}", True, WHITE)
            final_acc = big_font.render(f"Accuracy: {accuracy:.1f}%", True, CYAN)
            
            screen.blit(go_text, (WIDTH//2 - 140, HEIGHT//2 - 140))
            screen.blit(final_score, (WIDTH//2 - 100, HEIGHT//2 - 40))
            screen.blit(final_acc, (WIDTH//2 - 160, HEIGHT//2 + 30))
            
            if post_recording_frames <= 0:
                saved_text = font.render(f"Video Saved: {RECORD_FILENAME}", True, GREEN)
                restart_text = font.render("Press 'R' to Menu, 'Q' to Quit", True, WHITE)
                screen.blit(saved_text, (WIDTH//2 - 180, HEIGHT//2 + 80))
                screen.blit(restart_text, (WIDTH//2 - 140, HEIGHT//2 + 130))

        pygame.display.flip()
        
        should_record = False
        if game_state == "playing":
            should_record = True
        elif game_state == "game_over" and post_recording_frames > 0:
            should_record = True
            post_recording_frames -= 1
            if post_recording_frames == 0:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print("Recording Stopped (Score Screen Captured).")

        if should_record and video_writer is not None:
            view = pygame.surfarray.array3d(screen)
            view = view.transpose([1, 0, 2])
            img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    init_menu()
                if event.key == pygame.K_m and game_state == "playing":
                    if control_mode == "EYE":
                        control_mode = "HAND"
                    else:
                        control_mode = "EYE"
                    is_firing_gesture = False
                    pinch_states = {} 
                    select_fx.play()

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
