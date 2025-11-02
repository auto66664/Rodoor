from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import base64, cv2, numpy as np, mediapipe as mp
import time
import uuid

app = FastAPI()

# เปิด CORS ให้ HTML เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Global MediaPipe initialization (Performance optimization)
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Global variables สำหรับเก็บ session
sessions = {}

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

@app.get("/")
async def serve_html():
    # FileResponse requires that the file exists, using a placeholder for now
    return FileResponse("tutorial_Kaorop.html")

@app.post("/start_analysis")
async def start_analysis(request: Request):
    """เริ่ม session ใหม่"""
    # Note: data is unused here but retained for typical FastAPI structure
    data = await request.json() 
    session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "start_time": time.time(),
        "frame_count": 0,
        "total_similarity": 0,
        "calculation_done": False,
        "face_bonus": 0, # Default: 0 (No penalty)
        "face_bonus_given": False,
        "status": "processing"
    }
    
    print(f"🎯 Session started: {session_id}")
    return {"status": "started", "session_id": session_id}

@app.post("/analyze_pose")
async def analyze_pose(request: Request):
    """วิเคราะห์ pose จากภาพ"""
    try:
        data = await request.json()
        image_base64 = data["image"].split(",")[1]
        session_id = data.get("session_id")
        
        if not session_id or session_id not in sessions:
            return {"error": "Session not found", "status": "error"}
        
        session_data = sessions[session_id]
        
        if session_data["calculation_done"]:
            return {
                "status": "completed",
                "final_score": session_data.get("final_score", 0),
                "feedback": session_data.get("feedback", ""),
                "level": session_data.get("level", "")
            }
        
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Invalid image", "status": "error"}
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_elbow_angle = 45  
        target_hand_angle = 150  
        tolerance = 15
        average_percent = 0

        # ใช้ pose และ face_mesh ที่ประกาศไว้ข้างบน
        pose_results = pose.process(rgb)
        face_results = face_mesh.process(rgb)

        if not pose_results.pose_landmarks:
            print("❌ No pose detected in this frame!")

        # ✅ ตรวจการตะเบ๊ะ (แก้ไข logic เพื่อ "ลงโทษ" เมื่อมีการหันหน้า)
        if face_results.multi_face_landmarks and not session_data["face_bonus_given"]:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w, _ = frame.shape
                nose = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                nose_x = int(nose.x * w)
                left_x = int(left_eye.x * w)
                right_x = int(right_eye.x * w)

                diff_left = abs(nose_x - left_x)
                diff_right = abs(nose_x - right_x)
                diff_diff = abs(diff_left - diff_right)

                if diff_diff > 25 : #ตะเบ๊ะ
                    session_data["face_bonus"] = 0
                    session_data["face_bonus_given"] = True
                else:
                    session_data["face_bonus"] = -40
                break

        # ตรวจท่าทาง (ส่วนนี้เหมือนเดิม)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            try:
                rs = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                re = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                rw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                rt = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]

                elbow_angle = calculate_angle(rs, re, rw)
                hand_angle = calculate_angle(rt, rw, re)

                diff_elbow = abs(elbow_angle - target_elbow_angle)
                diff_hand = abs(hand_angle - target_hand_angle)

                diff_elbow = min(diff_elbow, tolerance)
                diff_hand = min(diff_hand, tolerance)

                similarity_elbow_pct = (1 - diff_elbow / tolerance) * 100
                similarity_hand_pct = (1 - diff_hand / tolerance) * 100

                average_percent = (similarity_elbow_pct + similarity_hand_pct) / 2
                
                print(f"📐 Angles - Elbow: {elbow_angle:.1f}°, Hand: {hand_angle:.1f}°, Score: {average_percent:.1f}%")

            except Exception as e:
                print(f"Error calculating angles: {e}")
                average_percent = 0

            session_data["frame_count"] += 1
            session_data["total_similarity"] += average_percent

        current_time = time.time()
        elapsed_time = current_time - session_data["start_time"]
        print(f"⏰ Elapsed time: {elapsed_time:.1f}s, Frames: {session_data['frame_count']}")

        if elapsed_time >= 7 and not session_data["calculation_done"]:
            if session_data["frame_count"] > 0:
                final_average = session_data["total_similarity"] / session_data["frame_count"]
                final_score = final_average + session_data["face_bonus"]
                final_score = max(0, min(100, final_score))
    
                feedback = ""
                # ปรับ Feedback ให้สอดคล้องกับการลงโทษการหันหน้า
                if session_data["face_bonus"] == -40:
                    feedback = "มึงลืมยกอกอึ๊บไอ่สัส!"

                if final_score < 20:
                    level = "กาก"
                elif final_score >= 20 and final_score < 50:
                    level = "พอใช้"
                elif final_score >= 50 and final_score < 80:
                    level = "ดี"
                elif final_score >= 80:
                    level = "โหดสัส"

                session_data.update({
                    "final_score": round(final_score, 1),
                    "feedback": feedback,
                    "level": level,
                    "calculation_done": True,
                    "status": "completed"
                })
    
                print(f"🎉 Analysis completed! Score: {final_score:.0f}%")
    
                return {
                    "status": "completed",
                    "final_score": session_data["final_score"],
                    "feedback": session_data["feedback"],
                    "level": session_data["level"]
                }
            else:
                session_data.update({
                    "final_score": 0,
                    "feedback": "ไม่พบท่าทางในช่วงเวลาที่กำหนด",
                    "level": "-",
                    "calculation_done": True,
                    "status": "completed"
                })
    
                return {
                    "status": "completed",
                    "final_score": 0,
                    "feedback": "ไม่พบท่าทางในช่วงเวลาที่กำหนด",
                    "level": "-"
                }

        return {"status": "waiting"}

    except Exception as e:
        print(f"❌ Error in analyze_pose: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/session_status/{session_id}")
async def get_session_status(session_id: str):
    """ตรวจสอบสถานะ session"""
    if session_id in sessions:
        return sessions[session_id]
    return {"error": "Session not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
