import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class GameDatabase:
    """Database manager for the waste sorting game"""
    
    def __init__(self, db_path: str = "waste_game.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                total_points INTEGER DEFAULT 0,
                level INTEGER DEFAULT 1,
                items_scanned INTEGER DEFAULT 0,
                correct_classifications INTEGER DEFAULT 0,
                wrong_classifications INTEGER DEFAULT 0,
                best_streak INTEGER DEFAULT 0,
                current_streak INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                last_active TIMESTAMP
            )
        ''')
        
        # Create game sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                points_earned INTEGER DEFAULT 0,
                items_scanned INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0.0,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create individual game actions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                waste_type TEXT,
                confidence REAL,
                selected_bin TEXT,
                correct_bin TEXT,
                is_correct BOOLEAN,
                points_awarded INTEGER,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES game_sessions (session_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create achievements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS achievements (
                achievement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                icon TEXT,
                requirement_type TEXT,
                requirement_value INTEGER,
                points_reward INTEGER
            )
        ''')
        
        # Create user achievements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_achievements (
                user_id TEXT,
                achievement_id INTEGER,
                earned_at TIMESTAMP,
                PRIMARY KEY (user_id, achievement_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (achievement_id) REFERENCES achievements (achievement_id)
            )
        ''')
        
        # Initialize default achievements
        self._init_default_achievements(cursor)
        
        conn.commit()
        conn.close()
    
    def _init_default_achievements(self, cursor):
        """Initialize default achievements"""
        default_achievements = [
            ("First Steps", "Scan your first item", "🎯", "items_scanned", 1, 10),
            ("Getting Started", "Scan 10 items", "🚀", "items_scanned", 10, 50),
            ("Explorer", "Scan 50 items", "🔍", "items_scanned", 50, 200),
            ("Expert Scanner", "Scan 100 items", "🏆", "items_scanned", 100, 500),
            ("Perfect Streak", "Get 5 correct in a row", "🔥", "best_streak", 5, 100),
            ("Streak Master", "Get 10 correct in a row", "⚡", "best_streak", 10, 300),
            ("Accuracy Pro", "Achieve 90% accuracy with 20+ items", "🎯", "accuracy_high", 90, 250),
            ("Point Collector", "Earn 1000 total points", "💎", "total_points", 1000, 100),
            ("Point Master", "Earn 5000 total points", "👑", "total_points", 5000, 500),
        ]
        
        for achievement in default_achievements:
            cursor.execute('''
                INSERT OR IGNORE INTO achievements 
                (name, description, icon, requirement_type, requirement_value, points_reward)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', achievement)
    
    def create_user(self, username: str) -> str:
        """Create a new user and return user_id"""
        user_id = f"user_{hashlib.md5(username.encode()).hexdigest()[:12]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (user_id, username, created_at, last_active)
                VALUES (?, ?, ?, ?)
            ''', (user_id, username, datetime.now(), datetime.now()))
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            # Username already exists, return existing user_id
            cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user information by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, total_points, level, items_scanned, 
                   correct_classifications, wrong_classifications, best_streak, 
                   current_streak, created_at, last_active
            FROM users WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'user_id': result[0],
                'username': result[1],
                'total_points': result[2],
                'level': result[3],
                'items_scanned': result[4],
                'correct_classifications': result[5],
                'wrong_classifications': result[6],
                'best_streak': result[7],
                'current_streak': result[8],
                'created_at': result[9],
                'last_active': result[10]
            }
        return None
    
    def start_game_session(self, user_id: str) -> str:
        """Start a new game session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[-6:]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO game_sessions (session_id, user_id, start_time)
            VALUES (?, ?, ?)
        ''', (session_id, user_id, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def record_action(self, session_id: str, user_id: str, waste_type: str, 
                     confidence: float, selected_bin: str, correct_bin: str) -> Dict:
        """Record a single classification action and return scoring result"""
        
        is_correct = selected_bin == correct_bin
        
        # Calculate points
        base_points = 10 if is_correct else -5
        confidence_bonus = 5 if confidence > 0.9 and is_correct else 0
        points_awarded = max(0, base_points + confidence_bonus)  # Minimum 0 points
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Record the action
        cursor.execute('''
            INSERT INTO game_actions 
            (session_id, user_id, waste_type, confidence, selected_bin, correct_bin, 
             is_correct, points_awarded, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_id, waste_type, confidence, selected_bin, correct_bin,
              is_correct, points_awarded, datetime.now()))
        
        # Update user statistics
        if is_correct:
            cursor.execute('''
                UPDATE users 
                SET total_points = total_points + ?,
                    items_scanned = items_scanned + 1,
                    correct_classifications = correct_classifications + 1,
                    current_streak = current_streak + 1,
                    best_streak = MAX(best_streak, current_streak + 1),
                    last_active = ?
                WHERE user_id = ?
            ''', (points_awarded, datetime.now(), user_id))
        else:
            cursor.execute('''
                UPDATE users 
                SET total_points = total_points + ?,
                    items_scanned = items_scanned + 1,
                    wrong_classifications = wrong_classifications + 1,
                    current_streak = 0,
                    last_active = ?
                WHERE user_id = ?
            ''', (points_awarded, datetime.now(), user_id))
        
        # Get updated user stats
        cursor.execute('SELECT current_streak, best_streak FROM users WHERE user_id = ?', (user_id,))
        streak_info = cursor.fetchone()
        current_streak, best_streak = streak_info if streak_info else (0, 0)
        
        conn.commit()
        conn.close()
        
        # Check for streak bonuses
        streak_bonus = 0
        if current_streak > 0 and current_streak % 5 == 0:  # Every 5 correct in a row
            streak_bonus = 20
            self.add_streak_bonus(user_id, streak_bonus)
        
        return {
            'is_correct': is_correct,
            'points_awarded': points_awarded,
            'streak_bonus': streak_bonus,
            'current_streak': current_streak,
            'confidence_bonus': confidence_bonus > 0
        }
    
    def add_streak_bonus(self, user_id: str, bonus_points: int):
        """Add streak bonus points"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET total_points = total_points + ?
            WHERE user_id = ?
        ''', (bonus_points, user_id))
        
        conn.commit()
        conn.close()
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top users leaderboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, total_points, level, items_scanned, 
                   CASE WHEN items_scanned > 0 
                        THEN ROUND(correct_classifications * 100.0 / items_scanned, 1)
                        ELSE 0 END as accuracy_rate
            FROM users 
            ORDER BY total_points DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        leaderboard = []
        for i, result in enumerate(results, 1):
            leaderboard.append({
                'rank': i,
                'username': result[0],
                'total_points': result[1],
                'level': result[2],
                'items_scanned': result[3],
                'accuracy_rate': result[4]
            })
        
        return leaderboard
    
    def get_user_achievements(self, user_id: str) -> List[Dict]:
        """Get user's earned achievements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.name, a.description, a.icon, ua.earned_at
            FROM achievements a
            JOIN user_achievements ua ON a.achievement_id = ua.achievement_id
            WHERE ua.user_id = ?
            ORDER BY ua.earned_at DESC
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        achievements = []
        for result in results:
            achievements.append({
                'name': result[0],
                'description': result[1],
                'icon': result[2],
                'earned_at': result[3]
            })
        
        return achievements
    
    def check_and_award_achievements(self, user_id: str) -> List[Dict]:
        """Check if user has earned any new achievements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current user stats
        user = self.get_user_by_id(user_id)
        if not user:
            return []
        
        # Get unearned achievements
        cursor.execute('''
            SELECT achievement_id, name, description, icon, requirement_type, 
                   requirement_value, points_reward
            FROM achievements 
            WHERE achievement_id NOT IN (
                SELECT achievement_id FROM user_achievements WHERE user_id = ?
            )
        ''', (user_id,))
        
        unearned = cursor.fetchall()
        new_achievements = []
        
        for achievement in unearned:
            achievement_id, name, description, icon, req_type, req_value, points_reward = achievement
            earned = False
            
            # Check achievement conditions
            if req_type == "items_scanned" and user['items_scanned'] >= req_value:
                earned = True
            elif req_type == "best_streak" and user['best_streak'] >= req_value:
                earned = True
            elif req_type == "total_points" and user['total_points'] >= req_value:
                earned = True
            elif req_type == "accuracy_high":
                if user['items_scanned'] >= 20:
                    accuracy = (user['correct_classifications'] / user['items_scanned']) * 100
                    if accuracy >= req_value:
                        earned = True
            
            if earned:
                # Award achievement
                cursor.execute('''
                    INSERT INTO user_achievements (user_id, achievement_id, earned_at)
                    VALUES (?, ?, ?)
                ''', (user_id, achievement_id, datetime.now()))
                
                # Award points
                cursor.execute('''
                    UPDATE users SET total_points = total_points + ? WHERE user_id = ?
                ''', (points_reward, user_id))
                
                new_achievements.append({
                    'name': name,
                    'description': description,
                    'icon': icon,
                    'points_reward': points_reward
                })
        
        conn.commit()
        conn.close()
        
        return new_achievements
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user information by user_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, total_points, level, items_scanned, 
                   correct_classifications, wrong_classifications, best_streak, 
                   current_streak, created_at, last_active
            FROM users WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'user_id': result[0],
                'username': result[1],
                'total_points': result[2],
                'level': result[3],
                'items_scanned': result[4],
                'correct_classifications': result[5],
                'wrong_classifications': result[6],
                'best_streak': result[7],
                'current_streak': result[8],
                'created_at': result[9],
                'last_active': result[10]
            }
        return None