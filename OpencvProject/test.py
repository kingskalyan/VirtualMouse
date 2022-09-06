from flask import Flask,render_template,Response
import cv2
import mediapipe as mp
import pyautogui as p
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
def generate_frames():
    
    cv = cv2.VideoCapture(0)
    hand_sol = mp.solutions.hands.Hands()
    draw = mp.solutions.drawing_utils
    scr_wid,sc_heig = p.size()
    ind_y = 0
    
    while True:
        success, frame = cv.read()
        if not success:
            break
        else:
            
            frame = cv2.flip(frame,1)
            fheig,fwid,_=frame.shape
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            OUT = hand_sol.process(rgb)
            hands = OUT.multi_hand_landmarks
            if hands:
                for hand in hands:
                    draw.draw_landmarks(frame,hand)
                    land = hand.landmark
                    for id,la in enumerate(land):
                        x = int(la.x*fwid)
                        y = int(la.y*fheig)
                        if id == 8:
                            cv2.circle(frame,(x,y),10,(0,255,255))
                            ind_x = scr_wid/fwid*x
                            ind_y = sc_heig/fheig*y
                            p.moveTo(ind_x,ind_y)
                        if id == 4:
                            cv2.circle(frame,(x,y),10,(0,255,255))
                            thumb_x = scr_wid/fwid*x
                            thumb_y = sc_heig/fheig*y
                            if abs(ind_y - thumb_y)<10:
                                p.click()
                                p.sleep(1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()          
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)