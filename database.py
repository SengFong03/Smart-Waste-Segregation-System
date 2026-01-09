import sqlite3
import os
import base64
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

class WasteDetectionDB:
    """Database manager for waste detection results and images"""
    
    def __init__(self, db_path: str = "waste_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table (for tracking upload batches)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT,
                created_at TIMESTAMP,
                total_images INTEGER,
                total_detections INTEGER,
                confidence_threshold REAL,
                description TEXT
            )
        ''')
        
        # Create images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                session_id TEXT,
                filename TEXT,
                original_image BLOB,
                annotated_image BLOB,
                file_hash TEXT,
                file_size INTEGER,
                uploaded_at TIMESTAMP,
                detection_count INTEGER,
                status TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Create detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT,
                session_id TEXT,
                waste_type TEXT,
                confidence REAL,
                recyclable BOOLEAN,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                detected_at TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images (image_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id)')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_name: str, confidence_threshold: float, description: str = "") -> str:
        """Create a new detection session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(session_name.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (session_id, session_name, created_at, total_images, 
                                total_detections, confidence_threshold, description)
            VALUES (?, ?, ?, 0, 0, ?, ?)
        ''', (session_id, session_name, datetime.now(), confidence_threshold, description))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def save_image_result(self, session_id: str, filename: str, original_image_data: bytes,
                         annotated_image_data: bytes, detections: List[Dict], file_size: int) -> str:
        """Save image and detection results to database"""
        
        # Generate unique image ID
        file_hash = hashlib.md5(original_image_data).hexdigest()
        image_id = f"img_{file_hash[:16]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if image already exists
        cursor.execute('SELECT image_id FROM images WHERE file_hash = ?', (file_hash,))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return existing[0]  # Return existing image ID
        
        # Save image data
        cursor.execute('''
            INSERT INTO images (image_id, session_id, filename, original_image, annotated_image,
                              file_hash, file_size, uploaded_at, detection_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image_id, session_id, filename, original_image_data, annotated_image_data,
              file_hash, file_size, datetime.now(), len(detections), 
              'success' if detections else 'no_detection'))
        
        # Save individual detections
        for detection in detections:
            if detection.get('waste_type') not in ['No detection'] and 'Error' not in str(detection.get('waste_type', '')):
                cursor.execute('''
                    INSERT INTO detections (image_id, session_id, waste_type, confidence, recyclable,
                                          bbox_x1, bbox_y1, bbox_x2, bbox_y2, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (image_id, session_id, detection['waste_type'], detection['confidence'],
                      detection['recyclable'], detection.get('bbox_x1', 0), detection.get('bbox_y1', 0),
                      detection.get('bbox_x2', 0), detection.get('bbox_y2', 0), datetime.now()))
        
        # Update session statistics
        cursor.execute('''
            UPDATE sessions 
            SET total_images = total_images + 1,
                total_detections = total_detections + ?
            WHERE session_id = ?
        ''', (len([d for d in detections if d.get('waste_type') not in ['No detection'] and 'Error' not in str(d.get('waste_type', ''))]), session_id))
        
        conn.commit()
        conn.close()
        
        return image_id
    
    def get_sessions(self, limit: int = 50) -> List[Dict]:
        """Get list of all sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, session_name, created_at, total_images, total_detections, 
                   confidence_threshold, description
            FROM sessions 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'session_name': row[1],
                'created_at': row[2],
                'total_images': row[3],
                'total_detections': row[4],
                'confidence_threshold': row[5],
                'description': row[6]
            })
        
        conn.close()
        return sessions
    
    def get_session_images(self, session_id: str) -> List[Dict]:
        """Get all images for a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT image_id, filename, file_size, uploaded_at, detection_count, status
            FROM images 
            WHERE session_id = ?
            ORDER BY uploaded_at ASC
        ''', (session_id,))
        
        images = []
        for row in cursor.fetchall():
            images.append({
                'image_id': row[0],
                'filename': row[1],
                'file_size': row[2],
                'uploaded_at': row[3],
                'detection_count': row[4],
                'status': row[5]
            })
        
        conn.close()
        return images
    
    def get_image_data(self, image_id: str) -> Optional[Dict]:
        """Get image data and detections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get image data
        cursor.execute('''
            SELECT filename, original_image, annotated_image, detection_count, status
            FROM images 
            WHERE image_id = ?
        ''', (image_id,))
        
        image_row = cursor.fetchone()
        if not image_row:
            conn.close()
            return None
        
        # Get detections
        cursor.execute('''
            SELECT waste_type, confidence, recyclable, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM detections 
            WHERE image_id = ?
            ORDER BY confidence DESC
        ''', (image_id,))
        
        detections = []
        for row in cursor.fetchall():
            detections.append({
                'waste_type': row[0],
                'confidence': row[1],
                'recyclable': bool(row[2]),
                'bbox': [row[3], row[4], row[5], row[6]]
            })
        
        conn.close()
        
        return {
            'filename': image_row[0],
            'original_image': image_row[1],
            'annotated_image': image_row[2],
            'detection_count': image_row[3],
            'status': image_row[4],
            'detections': detections
        }
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete detections first
            cursor.execute('DELETE FROM detections WHERE session_id = ?', (session_id,))
            
            # Delete images
            cursor.execute('DELETE FROM images WHERE session_id = ?', (session_id,))
            
            # Delete session
            cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
            
            conn.commit()
            success = cursor.rowcount > 0
            
        except Exception as e:
            conn.rollback()
            success = False
            
        finally:
            conn.close()
        
        return success
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count sessions
        cursor.execute('SELECT COUNT(*) FROM sessions')
        total_sessions = cursor.fetchone()[0]
        
        # Count images
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]
        
        # Count detections
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]
        
        # Get database file size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        # Get waste type distribution
        cursor.execute('''
            SELECT waste_type, COUNT(*) as count 
            FROM detections 
            GROUP BY waste_type 
            ORDER BY count DESC
        ''')
        waste_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_sessions': total_sessions,
            'total_images': total_images,
            'total_detections': total_detections,
            'database_size_mb': db_size / (1024 * 1024),
            'waste_distribution': waste_distribution
        }