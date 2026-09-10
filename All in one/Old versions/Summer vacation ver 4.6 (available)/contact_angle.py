import cv2, numpy as np
def precise_contact_angle(image_path="drop_image.jpg"):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("无法读取图像")
    gray, blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3,3), 0)
    points = []    
    def mouse_cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x,y)); print(f"点{len(points)}: ({x},{y})")
            cv2.circle(img, (x,y), 5, (0,255,0), -1); cv2.imshow("Image", img)
            if len(points) == 2: process()    
    def extract_edge(cp, side):
        cx,cy,h,w = *cp,60,50; ys,ye = max(0,cy-h), cy
        xs,xe = (max(0,cx-w), min(img.shape[1],cx+20)) if side=='left' else (max(0,cx-20), min(img.shape[1],cx+w))
        roi = blur[ys:ye,xs:xe]; ep = []
        for yi,r in enumerate(roi):
            g = np.diff(r.astype(float))
            if (np.max(g) if side=='left' else np.max(-g)) > 15:
                ep.append((xs + np.argmax(g if side=='left' else -g), ys+yi))
        return np.array(ep) if ep else None    
    def calc_angle(pts, side):
        if pts is None or len(pts)<5: return 0,None
        try:
            a,b,c = np.polyfit(pts[:,1], pts[:,0], 2); yc = np.max(pts[:,1])
            tv, bv = np.array([-2*a*yc-b, -1]), np.array([1,0] if side=='left' else [-1,0])
            return np.degrees(np.arccos(np.clip(np.dot(tv,bv)/(np.linalg.norm(tv)*np.linalg.norm(bv)), -1,1))), (a,b,c)
        except Exception as e: print(f"错误: {e}"); return 0,None    
    def process():
        p1,p2 = points; pl,pr = extract_edge(p1,'left'), extract_edge(p2,'right')
        al,cl = calc_angle(pl,'left'); ar,cr = calc_angle(pr,'right')
        print(f"左:{al:.1f}° 右:{ar:.1f}° 平均:{(al+ar)/2:.1f}°"); draw(p1,p2,pl,pr,cl,cr,al,ar)    
    def draw(p1,p2,pl,pr,cl,cr,al,ar):
        vis = img.copy(); cv2.line(vis,p1,p2,(255,0,0),2)
        for pts in (pl,pr):
            if pts is not None: [cv2.circle(vis,(x,y),3,(0,255,0),-1) for x,y in pts]
        for pts,coeff in ((pl,cl),(pr,cr)):
            if pts is not None and coeff is not None:
                a,b,c = coeff; ys = np.linspace(np.min(pts[:,1]),np.max(pts[:,1]),50)
                cv2.polylines(vis,[np.column_stack(((a*ys**2+b*ys+c).astype(int),ys.astype(int)))],False,(0,255,255),2)
        cv2.putText(vis,f"L:{al:.1f}°",(p1[0]-60,p1[1]-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.putText(vis,f"R:{ar:.1f}°",(p2[0]+10,p2[1]-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("Result",vis); cv2.imwrite("result_precise.jpg",vis); print("已保存。按 'q' 退出。")  
    cv2.namedWindow("Image"); cv2.setMouseCallback("Image", mouse_cb); cv2.imshow("Image", img)
    while cv2.waitKey(1)&0xFF != ord('q'): pass
    cv2.destroyAllWindows()
if __name__ == "__main__": precise_contact_angle()
