"""
Agar.io Simulation - Flask Visualization (Cross-Compatible Version)
A single-file Flask application that:
- Runs the simulated Agar.io game (player vs bots)
- Trains DQN/NFSP agents in real-time with up to 2000x speed
- Configurable target episodes via UI
- Visualizes everything in the browser
- Models are cross-compatible with the live Agar.io game
"""

from flask import Flask, render_template_string, jsonify
import threading
import time
import os

from game_env import AgarioSimEnv
from dqn_agent import DQNAgent
from nfsp_agent import NFSPAgent

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)

# Global game state
game_state = {
    'env': None,
    'agent': None,
    'training': False,
    'paused': False,
    'speed': 10.0,
    'mode': 'dqn',  # 'dqn', 'nfsp', or 'manual'
    'target_episodes': 5000,
    'manual_action': 4,  # Default action for manual mode (stay still/up-right)
}


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agar.io Trainer - DQN/NFSP</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --accent: #00ff88;
            --accent-dim: #00cc6a;
            --danger: #ff3366;
            --warning: #ffaa00;
            --text: #e0e0e0;
            --text-dim: #888;
            --border: #2a2a3a;
        }
        
        body {
            font-family: 'Space Mono', monospace;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(0, 255, 136, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 51, 102, 0.03) 0%, transparent 50%),
                linear-gradient(135deg, #0a0a0f 0%, #151520 100%);
            z-index: -1;
        }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
        }
        
        h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            background: linear-gradient(135deg, var(--accent) 0%, #00ccff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 4px;
        }
        
        .subtitle { color: var(--text-dim); font-size: 0.85rem; letter-spacing: 2px; margin-top: 5px; }
        
        .main-grid { display: grid; grid-template-columns: 1fr 320px; gap: 20px; }
        
        .game-container {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid var(--border);
        }
        
        .game-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        
        .game-title { font-family: 'Orbitron', sans-serif; font-size: 1rem; color: var(--accent); }
        
        .badge {
            padding: 6px 12px;
            border-radius: 15px;
            font-weight: 700;
            font-size: 0.75rem;
        }
        
        .badge-episode {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%);
            color: var(--bg-dark);
        }
        
        .badge-mode {
            background: rgba(0, 204, 255, 0.2);
            color: #00ccff;
            border: 1px solid rgba(0, 204, 255, 0.3);
        }
        
        #gameCanvas {
            width: 100%;
            height: auto;
            border-radius: 6px;
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .controls { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
        
        .btn {
            font-family: 'Space Mono', monospace;
            padding: 10px 18px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 1px;
            transition: all 0.2s ease;
            text-transform: uppercase;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%);
            color: var(--bg-dark);
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3); }
        
        .btn-secondary {
            background: var(--bg-dark);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--danger) 0%, #cc2952 100%);
            color: white;
        }
        
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 12px;
            padding: 10px;
            background: var(--bg-dark);
            border-radius: 6px;
        }
        
        .speed-control label { font-size: 0.75rem; color: var(--text-dim); }
        
        .speed-control input[type="range"] {
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            background: #2a2a3a;
            border-radius: 3px;
        }
        
        .speed-control input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px; height: 14px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .speed-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1rem;
            color: var(--accent);
            min-width: 50px;
            text-align: right;
        }
        
        .sidebar { display: flex; flex-direction: column; gap: 15px; }
        
        .card {
            background: var(--bg-card);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid var(--border);
        }
        
        .card-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            color: var(--accent);
            margin-bottom: 12px;
            letter-spacing: 2px;
        }
        
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        
        .stat-item {
            text-align: center;
            padding: 12px 8px;
            background: rgba(0, 255, 136, 0.05);
            border-radius: 6px;
            border: 1px solid rgba(0, 255, 136, 0.1);
        }
        
        .stat-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--accent);
        }
        
        .stat-label {
            font-size: 0.65rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        }
        
        .progress-item { margin-bottom: 12px; }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 0.75rem;
        }
        
        .progress-bar {
            height: 6px;
            background: var(--bg-dark);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent) 0%, #00ccff 100%);
            transition: width 0.3s ease;
        }
        
        .progress-fill.danger { background: linear-gradient(90deg, var(--danger) 0%, #ff6699 100%); }
        .progress-fill.warning { background: linear-gradient(90deg, var(--warning) 0%, #ffcc00 100%); }
        
        .kd-row { display: flex; gap: 10px; margin-top: 8px; }
        
        .kd-item {
            flex: 1;
            text-align: center;
            padding: 8px;
            border-radius: 6px;
        }
        
        .kd-item.kills { background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.2); }
        .kd-item.deaths { background: rgba(255, 51, 102, 0.1); border: 1px solid rgba(255, 51, 102, 0.2); }
        
        .kd-value { font-family: 'Orbitron', sans-serif; font-size: 1.5rem; font-weight: 700; }
        .kd-item.kills .kd-value { color: var(--accent); }
        .kd-item.deaths .kd-value { color: var(--danger); }
        .kd-label { font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
        }
        
        .status.training { background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); color: var(--accent); }
        .status.paused { background: rgba(255, 170, 0, 0.1); border: 1px solid rgba(255, 170, 0, 0.3); color: var(--warning); }
        .status.stopped { background: rgba(255, 51, 102, 0.1); border: 1px solid rgba(255, 51, 102, 0.3); color: var(--danger); }
        
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        
        .status.training .status-dot { background: var(--accent); }
        .status.paused .status-dot { background: var(--warning); animation: none; }
        .status.stopped .status-dot { background: var(--danger); animation: none; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.8); }
        }
        
        .chart-container {
            height: 120px;
            background: var(--bg-dark);
            border-radius: 6px;
            padding: 8px;
        }
        
        #rewardChart { width: 100%; height: 100%; }
        
        .mode-select {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }
        
        .mode-btn {
            flex: 1;
            padding: 8px;
            border: 1px solid var(--border);
            background: var(--bg-dark);
            color: var(--text-dim);
            border-radius: 6px;
            cursor: pointer;
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            transition: all 0.2s;
        }
        
        .mode-btn.active {
            background: rgba(0, 255, 136, 0.1);
            border-color: var(--accent);
            color: var(--accent);
        }
        
        footer {
            text-align: center;
            padding: 20px;
            margin-top: 20px;
            border-top: 1px solid var(--border);
            color: var(--text-dim);
            font-size: 0.75rem;
        }
        
        @media (max-width: 1100px) {
            .main-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AGAR.IO TRAINER</h1>
            <p class="subtitle">DQN + NFSP + MANUAL PLAY • CROSS-COMPATIBLE • UP TO 2000X SPEED</p>
        </header>
        
        <div class="main-grid">
            <div class="game-container">
                <div class="game-header">
                    <span class="game-title">SIMULATION</span>
                    <div style="display: flex; gap: 8px;">
                        <span class="badge badge-mode" id="modeBadge">DQN</span>
                        <span class="badge badge-episode" id="episodeBadge">EP 0</span>
                    </div>
                </div>
                <canvas id="gameCanvas" width="800" height="533"></canvas>
                
                <div class="mode-select">
                    <button class="mode-btn active" id="dqnBtn" onclick="setMode('dqn')">DQN</button>
                    <button class="mode-btn" id="nfspBtn" onclick="setMode('nfsp')">NFSP</button>
                    <button class="mode-btn" id="manualBtn" onclick="setMode('manual')">MANUAL</button>
                </div>
                
                <div id="manualControls" style="display: none; margin-top: 10px; padding: 10px; background: var(--bg-dark); border-radius: 6px; text-align: center;">
                    <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 8px;">USE ARROW KEYS OR WASD TO MOVE</div>
                    <div style="display: grid; grid-template-columns: repeat(3, 40px); gap: 4px; justify-content: center; margin-bottom: 8px;">
                        <div></div>
                        <button class="btn btn-secondary" style="padding: 8px; font-size: 0.7rem;" onclick="manualMove(0)">↑</button>
                        <div></div>
                        <button class="btn btn-secondary" style="padding: 8px; font-size: 0.7rem;" onclick="manualMove(2)">←</button>
                        <button class="btn btn-secondary" style="padding: 8px; font-size: 0.7rem;" onclick="manualMove(4)">⬉</button>
                        <button class="btn btn-secondary" style="padding: 8px; font-size: 0.7rem;" onclick="manualMove(3)">→</button>
                        <div></div>
                        <button class="btn btn-secondary" style="padding: 8px; font-size: 0.7rem;" onclick="manualMove(1)">↓</button>
                        <div></div>
                    </div>
                    <div style="font-size: 0.7rem; color: var(--text-dim);">Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4-7=Diagonals</div>
                </div>
                
                <div class="controls">
                    <button class="btn btn-primary" id="startBtn" onclick="startTraining()">START</button>
                    <button class="btn btn-secondary" id="pauseBtn" onclick="togglePause()" disabled>PAUSE</button>
                    <button class="btn btn-danger" id="resetBtn" onclick="resetTraining()">RESET</button>
                    <button class="btn btn-secondary" onclick="saveModel()">SAVE MODEL</button>
                </div>
                
                <div class="speed-control">
                    <label>SPEED:</label>
                    <input type="range" id="speedSlider" min="1" max="5000" value="100" oninput="updateSpeed(this.value)">
                    <span class="speed-value" id="speedLabel">100x</span>
                </div>
                <div class="speed-control">
                    <label>TARGET EPS:</label>
                    <input type="number" id="targetEpisodes" value="5000" min="100" max="100000" style="width: 80px; background: var(--bg-dark); border: 1px solid var(--border); color: var(--text); padding: 4px 8px; border-radius: 4px; font-family: 'Orbitron', sans-serif;">
                    <button class="btn btn-secondary" onclick="setTargetEpisodes()" style="padding: 6px 12px; font-size: 0.7rem;">SET</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <div class="card-title">STATUS</div>
                    <div class="status stopped" id="statusIndicator">
                        <span class="status-dot"></span>
                        <span id="statusText">READY</span>
                    </div>
                    <div class="kd-row">
                        <div class="kd-item kills">
                            <div class="kd-value" id="totalKills">0</div>
                            <div class="kd-label">Kills</div>
                        </div>
                        <div class="kd-item deaths">
                            <div class="kd-value" id="totalDeaths">0</div>
                            <div class="kd-label">Deaths</div>
                        </div>
                    </div>
                    <div class="kd-row" style="margin-top: 8px;">
                        <div class="kd-item" style="background: rgba(0, 204, 255, 0.1); border: 1px solid rgba(0, 204, 255, 0.2);">
                            <div class="kd-value" id="totalWins" style="color: #00ccff;">0</div>
                            <div class="kd-label">Wins</div>
                        </div>
                        <div class="kd-item" style="background: rgba(255, 170, 0, 0.1); border: 1px solid rgba(255, 170, 0, 0.2);">
                            <div class="kd-value" id="winRate" style="color: #ffaa00;">0%</div>
                            <div class="kd-label">Win Rate</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">TRAINING</div>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="statEpisode">0</div>
                            <div class="stat-label">Episodes</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="statSteps">0</div>
                            <div class="stat-label">Steps</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="statReward">0</div>
                            <div class="stat-label">Avg Reward</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="statEpsilon">1.00</div>
                            <div class="stat-label">Epsilon</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">AGENT</div>
                    <div class="progress-item">
                        <div class="progress-header">
                            <span>Exploration</span>
                            <span id="explorationValue">100%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="explorationBar" style="width: 100%"></div>
                        </div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-header">
                            <span>Player Size</span>
                            <span id="playerSizeValue">25</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill warning" id="playerSizeBar" style="width: 25%"></div>
                        </div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-header">
                            <span>Training Progress</span>
                            <span id="progressValue">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressBar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">REWARD HISTORY</div>
                    <div class="chart-container">
                        <canvas id="rewardChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            CS238 RL Project | State: 24-dim (cross-compatible) | Actions: 8 directions
        </footer>
    </div>
    
    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        
        let rewardHistory = [];
        const maxHistory = 100;
        let pollInterval = null;
        let currentMode = 'dqn';
        
        function setMode(mode) {
            currentMode = mode;
            document.getElementById('dqnBtn').classList.toggle('active', mode === 'dqn');
            document.getElementById('nfspBtn').classList.toggle('active', mode === 'nfsp');
            document.getElementById('manualBtn').classList.toggle('active', mode === 'manual');
            document.getElementById('modeBadge').textContent = mode.toUpperCase();
            
            // Show/hide manual controls
            const manualControls = document.getElementById('manualControls');
            if (mode === 'manual') {
                manualControls.style.display = 'block';
                document.getElementById('speedSlider').disabled = false;
                document.getElementById('speedSlider').max = '100';
                document.getElementById('speedSlider').value = '10';
                updateSpeed(10);
            } else {
                manualControls.style.display = 'none';
                document.getElementById('speedSlider').disabled = false;
                document.getElementById('speedSlider').max = '5000';
            }
            
            fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            });
        }
        
        function manualMove(action) {
            fetch('/api/manual_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            });
        }
        
        // Keyboard controls for manual mode
        document.addEventListener('keydown', function(e) {
            if (currentMode !== 'manual' || !document.getElementById('statusIndicator').classList.contains('training')) {
                return;
            }
            
            const keyMap = {
                'ArrowUp': 0, 'w': 0, 'W': 0,
                'ArrowDown': 1, 's': 1, 'S': 1,
                'ArrowLeft': 2, 'a': 2, 'A': 2,
                'ArrowRight': 3, 'd': 3, 'D': 3,
                'q': 6, 'Q': 6,  // Up-Left
                'e': 4, 'E': 4,  // Up-Right
                'z': 7, 'Z': 7,  // Down-Left
                'c': 5, 'C': 5   // Down-Right
            };
            
            if (e.key in keyMap) {
                e.preventDefault();
                manualMove(keyMap[e.key]);
            }
        });
        
        function startTraining() {
            if (currentMode === 'manual') {
                document.getElementById('startBtn').textContent = 'PLAY';
            }
            
            fetch('/api/start', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'started') {
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('pauseBtn').disabled = false;
                        updateStatus('training');
                        if (!pollInterval) pollInterval = setInterval(updateGame, 50);
                    }
                });
        }
        
        function togglePause() {
            fetch('/api/pause', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('pauseBtn').textContent = data.paused ? 'RESUME' : 'PAUSE';
                    updateStatus(data.paused ? 'paused' : 'training');
                });
        }
        
        function resetTraining() {
            fetch('/api/reset', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('pauseBtn').disabled = true;
                    document.getElementById('pauseBtn').textContent = 'PAUSE';
                    document.getElementById('startBtn').textContent = currentMode === 'manual' ? 'PLAY' : 'START';
                    updateStatus('stopped');
                    rewardHistory = [];
                    drawChart();
                });
        }
        
        function saveModel() {
            fetch('/api/save', { method: 'POST' })
                .then(r => r.json())
                .then(data => alert('Model saved: ' + data.path));
        }
        
        function updateSpeed(val) {
            document.getElementById('speedLabel').textContent = val + 'x';
            fetch('/api/speed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speed: parseFloat(val) })
            });
        }
        
        function setTargetEpisodes() {
            const val = parseInt(document.getElementById('targetEpisodes').value) || 5000;
            fetch('/api/target_episodes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target: val })
            }).then(r => r.json()).then(data => {
                targetEps = data.target;
                alert('Target episodes set to: ' + data.target);
            });
        }
        
        let targetEps = 5000;
        
        function updateStatus(status) {
            const el = document.getElementById('statusIndicator');
            const txt = document.getElementById('statusText');
            el.className = 'status ' + status;
            txt.textContent = status === 'training' ? 'TRAINING' : status === 'paused' ? 'PAUSED' : 'READY';
        }
        
        function updateGame() {
            fetch('/api/state').then(r => r.json()).then(data => {
                if (data.game_state) drawGame(data.game_state);
                if (data.stats) updateStats(data.stats);
                if (data.game_state) updateKD(data.game_state);
            });
        }
        
        function drawGame(state) {
            const scaleX = canvas.width / state.game_info.width;
            const scaleY = canvas.height / state.game_info.height;
            
            // Background
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Grid
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            for (let x = 0; x < canvas.width; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
            for (let y = 0; y < canvas.height; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
            
            // Food particles (small dots)
            if (state.food) {
                state.food.forEach(f => {
                    ctx.beginPath();
                    ctx.arc(f.x * scaleX, f.y * scaleY, Math.max(f.radius * scaleX, 2), 0, Math.PI * 2);
                    ctx.fillStyle = f.color;
                    ctx.fill();
                });
            }
            
            // Bots (varying sizes and behaviors)
            if (state.bots) {
                state.bots.forEach(b => {
                    if (!b.alive) return;
                    const bx = b.x * scaleX;
                    const by = b.y * scaleY;
                    const br = b.radius * scaleX;
                    
                    // Glow intensity based on size
                    ctx.shadowColor = b.color;
                    ctx.shadowBlur = b.radius > 40 ? 15 : 8;
                    
                    ctx.beginPath();
                    ctx.arc(bx, by, br, 0, Math.PI * 2);
                    ctx.fillStyle = b.color;
                    ctx.fill();
                    
                    // Inner highlight
                    ctx.beginPath();
                    ctx.arc(bx - br * 0.25, by - br * 0.25, br * 0.2, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(255,255,255,0.2)';
                    ctx.fill();
                    
                    ctx.shadowBlur = 0;
                    
                    // Size label for larger bots
                    if (b.radius > 25) {
                        ctx.fillStyle = '#fff';
                        ctx.font = 'bold 8px Orbitron';
                        ctx.textAlign = 'center';
                        ctx.fillText(Math.round(b.radius), bx, by + 3);
                    }
                });
            }
            
            // Player
            if (state.player && state.player.alive) {
                const p = state.player;
                ctx.beginPath();
                ctx.arc(p.x * scaleX, p.y * scaleY, p.radius * scaleX, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.shadowColor = p.color;
                ctx.shadowBlur = 12;
                ctx.fill();
                ctx.shadowBlur = 0;
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 10px Orbitron';
                ctx.textAlign = 'center';
                ctx.fillText(Math.round(p.radius), p.x * scaleX, p.y * scaleY + 4);
            }
            
            document.getElementById('episodeBadge').textContent = 'EP ' + state.game_info.episode;
        }
        
        function updateKD(state) {
            document.getElementById('totalKills').textContent = state.game_info.total_kills || 0;
            document.getElementById('totalDeaths').textContent = state.game_info.total_deaths || 0;
            // Wins are updated from stats, not game_info
        }
        
        function updateStats(stats) {
            document.getElementById('statEpisode').textContent = stats.episodes || 0;
            document.getElementById('statSteps').textContent = formatNum(stats.steps || stats.total_steps || 0);
            document.getElementById('statReward').textContent = (stats.avg_reward_10 || 0).toFixed(1);
            
            // Handle epsilon for DQN/NFSP
            if (stats.epsilon !== undefined) {
                document.getElementById('statEpsilon').textContent = (stats.epsilon).toFixed(2);
            } else {
                document.getElementById('statEpsilon').textContent = '-';
            }
            
            // Wins display
            document.getElementById('totalWins').textContent = stats.total_wins || 0;
            document.getElementById('winRate').textContent = (stats.win_rate || 0).toFixed(1) + '%';
            
            // Exploration bar - use epsilon for DQN/NFSP
            const eps = stats.epsilon !== undefined ? stats.epsilon : 0.5;
            document.getElementById('explorationValue').textContent = (eps * 100).toFixed(0) + '%';
            document.getElementById('explorationBar').style.width = (eps * 100) + '%';
            
            // Progress
            const progress = Math.min((stats.episodes || 0) / targetEps * 100, 100);
            document.getElementById('progressValue').textContent = progress.toFixed(1) + '% (' + (stats.episodes || 0) + '/' + targetEps + ')';
            document.getElementById('progressBar').style.width = progress + '%';
            
            if (stats.last_reward !== undefined && stats.episodes > rewardHistory.length) {
                rewardHistory.push(stats.last_reward);
                if (rewardHistory.length > maxHistory) rewardHistory.shift();
                drawChart();
            }
        }
        
        function drawChart() {
            const c = document.getElementById('rewardChart');
            const cx = c.getContext('2d');
            c.width = c.parentElement.clientWidth - 16;
            c.height = 104;
            
            cx.fillStyle = '#0a0a0f';
            cx.fillRect(0, 0, c.width, c.height);
            
            if (rewardHistory.length < 2) return;
            
            const min = Math.min(...rewardHistory);
            const max = Math.max(...rewardHistory);
            const range = max - min || 1;
            
            cx.beginPath();
            cx.strokeStyle = '#00ff88';
            cx.lineWidth = 2;
            
            const stepX = c.width / (rewardHistory.length - 1);
            rewardHistory.forEach((r, i) => {
                const x = i * stepX;
                const y = c.height - ((r - min) / range) * (c.height - 16) - 8;
                if (i === 0) cx.moveTo(x, y);
                else cx.lineTo(x, y);
            });
            cx.stroke();
            
            cx.lineTo(c.width, c.height);
            cx.lineTo(0, c.height);
            cx.closePath();
            cx.fillStyle = 'rgba(0, 255, 136, 0.1)';
            cx.fill();
        }
        
        function formatNum(n) {
            if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
            if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
            return n;
        }
        
        // Init
        drawGame({ bots: [], food: [], player: null, game_info: { episode: 0, width: 3000, height: 2000, total_kills: 0, total_deaths: 0 } });
    </script>
</body>
</html>
'''


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    if game_state['env'] is None:
        return jsonify({'game_state': None, 'stats': {}})
    
    env = game_state['env']
    
    # Get stats based on mode
    if game_state['mode'] == 'manual':
        # Return manual play stats
        stats = getattr(env, 'manual_stats', {
            'episodes': 0,
            'total_steps': 0,
            'total_reward': 0,
            'total_kills': 0,
            'total_deaths': 0,
            'total_wins': 0,
            'last_reward': 0,
        })
        # Add computed fields for UI
        stats['steps'] = stats.get('total_steps', 0)
        stats['avg_reward_10'] = stats.get('last_reward', 0)
        stats['win_rate'] = (stats.get('total_wins', 0) / max(1, stats.get('episodes', 1))) * 100
    else:
        stats = game_state['agent'].get_stats() if game_state['agent'] else {}
    
    return jsonify({
        'game_state': env.get_game_state(),
        'stats': stats
    })


@app.route('/api/start', methods=['POST'])
def start_training():
    if game_state['training']:
        return jsonify({'status': 'already_running'})
    
    game_state['training'] = True
    game_state['paused'] = False
    
    thread = threading.Thread(target=training_loop, daemon=True)
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/api/pause', methods=['POST'])
def pause_training():
    game_state['paused'] = not game_state['paused']
    return jsonify({'paused': game_state['paused']})


@app.route('/api/reset', methods=['POST'])
def reset_training():
    game_state['training'] = False
    game_state['paused'] = False
    time.sleep(0.2)
    
    game_state['env'] = AgarioSimEnv()
    
    if game_state['mode'] == 'manual':
        # Manual mode doesn't need an agent
        game_state['agent'] = None
    elif game_state['mode'] == 'nfsp':
        game_state['agent'] = NFSPAgent(
            state_dim=game_state['env'].state_dim,
            action_dim=game_state['env'].action_dim
        )
    else:
        game_state['agent'] = DQNAgent(
            state_dim=game_state['env'].state_dim,
            action_dim=game_state['env'].action_dim
        )
    
    return jsonify({'status': 'reset'})


@app.route('/api/speed', methods=['POST'])
def set_speed():
    from flask import request
    data = request.get_json()
    game_state['speed'] = float(data.get('speed', 1.0))
    return jsonify({'speed': game_state['speed']})


@app.route('/api/target_episodes', methods=['POST'])
def set_target_episodes():
    from flask import request
    data = request.get_json()
    game_state['target_episodes'] = int(data.get('target', 5000))
    return jsonify({'target': game_state['target_episodes']})


@app.route('/api/mode', methods=['POST'])
def set_mode():
    from flask import request
    data = request.get_json()
    game_state['mode'] = data.get('mode', 'dqn')
    return jsonify({'mode': game_state['mode']})


@app.route('/api/manual_action', methods=['POST'])
def set_manual_action():
    from flask import request
    data = request.get_json()
    game_state['manual_action'] = int(data.get('action', 4))
    return jsonify({'action': game_state['manual_action']})


@app.route('/api/save', methods=['POST'])
def save_model():
    if game_state['agent']:
        path = f"{game_state['mode']}_model.pth"
        game_state['agent'].save(path)
        
        # Also export for live game
        if hasattr(game_state['agent'], 'export_for_live_game'):
            game_state['agent'].export_for_live_game("live_model.pth")
        
        return jsonify({'status': 'saved', 'path': path})
    return jsonify({'status': 'no_agent'})


# ============================================================================
# TRAINING LOOP
# ============================================================================

def training_loop():
    if game_state['env'] is None:
        game_state['env'] = AgarioSimEnv()
    
    # Manual mode doesn't need an agent
    if game_state['mode'] == 'manual':
        manual_play_loop()
        return
    
    if game_state['agent'] is None:
        if game_state['mode'] == 'nfsp':
            game_state['agent'] = NFSPAgent(
                state_dim=game_state['env'].state_dim,
                action_dim=game_state['env'].action_dim
            )
        else:
            game_state['agent'] = DQNAgent(
                state_dim=game_state['env'].state_dim,
                action_dim=game_state['env'].action_dim
            )
    
    env = game_state['env']
    agent = game_state['agent']
    
    # NFSP episode begin
    if hasattr(agent, 'begin_episode'):
        agent.begin_episode()
    
    state = env.reset()
    episode_reward = 0
    
    while game_state['training']:
        if game_state['paused']:
            time.sleep(0.1)
            continue
        
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done)
        if hasattr(agent, 'learn'):
            agent.learn()
        
        episode_reward += reward
        state = next_state
        
        if done:
            # Check if this was a win (player still alive = won)
            won = info.get('won', False)
            kills = info.get('episode_kills', 0)
            agent.end_episode(episode_reward, won=won, kills=kills)
            
            # NFSP episode begin
            if hasattr(agent, 'begin_episode'):
                agent.begin_episode()
            
            state = env.reset()
            episode_reward = 0
        
        # Speed control
        delay = 0.01 / game_state['speed']
        if delay > 0.0001:
            time.sleep(delay)


def manual_play_loop():
    """Manual play mode - user controls the player"""
    env = game_state['env']
    state = env.reset()
    episode_reward = 0
    steps = 0
    
    # Create a simple stats tracker for manual mode
    manual_stats = {
        'episodes': 0,
        'total_steps': 0,
        'total_reward': 0,
        'total_kills': 0,
        'total_deaths': 0,
        'total_wins': 0,
        'last_reward': 0,
    }
    
    while game_state['training']:
        if game_state['paused']:
            time.sleep(0.1)
            continue
        
        # Get action from user input
        action = game_state['manual_action']
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        state = next_state
        
        if done:
            # Update manual stats
            manual_stats['episodes'] += 1
            manual_stats['total_steps'] += steps
            manual_stats['total_reward'] += episode_reward
            manual_stats['last_reward'] = episode_reward
            manual_stats['total_kills'] = env.total_kills
            manual_stats['total_deaths'] = env.total_deaths
            
            won = info.get('won', False)
            if won:
                manual_stats['total_wins'] += 1
            
            # Store stats in env for display
            if not hasattr(env, 'manual_stats'):
                env.manual_stats = manual_stats
            else:
                env.manual_stats.update(manual_stats)
            
            # Reset for next episode
            state = env.reset()
            episode_reward = 0
            steps = 0
        
        # Speed control (manual mode is slower by default)
        delay = 0.01 / game_state['speed']
        if delay > 0.0001:
            time.sleep(delay)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("AGAR.IO TRAINER")
    print("Cross-compatible with live game")
    print("=" * 60)
    print("Modes:")
    print("  - DQN (value-based RL)")
    print("  - NFSP (self-play RL)")
    print("  - MANUAL (play yourself!)")
    print("Open http://localhost:3000")
    print("=" * 60)
    
    game_state['env'] = AgarioSimEnv()
    game_state['agent'] = DQNAgent(
        state_dim=game_state['env'].state_dim,
        action_dim=game_state['env'].action_dim
    )
    
    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)
