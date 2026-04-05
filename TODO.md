# Traffic Surveillance Redesign TODO
## Approved Plan Implementation Steps

### 1. Create/Update core files structure [IN PROGRESS]
- [x] Create TODO.md ✅
- [x] Edit app.py: Redesign to image/video only, tabs, remove live cams (initial cleanup done)
- [ ] Edit detector.py: Cleanup, integrate utils drawing
- [x] Edit utils.py: Simplify config, keep drawing/stats
- [x] Update requirements.txt: Verified deps (no changes needed)
- [ ] Update README.md: New simple description
- [x] Delete traffic_signs.py (unused)

### 2. Test core functionality [TODO]
- [ ] streamlit run app.py
- [ ] Test image upload → detections → annotated + stats
- [ ] Test video upload → all frames process → aggregate summary/export

### 3. Polish & complete [TODO]
- [ ] Add JSON/CSV export for detections
- [ ] Verify YOLO detects vehicles/signs/lights (COCO classes)
- [ ] attempt_completion

**Next step: Edit app.py**

