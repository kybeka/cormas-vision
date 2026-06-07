# CORMAS Vision Project Timeline

Project planning and milestone schedule for computer vision integration with CORMAS board game simulations.

## Project Schedule

| Week | Date Range | Phase | Primary Focus | Weekly Plan |
|------|------------|-------|---------------|--------------|
| 1 | Jun 26 - Jul 2 | Setup | Repository & initial research | • Set up GitHub repository and connect to Pharo via Iceberg<br>• Extract frames from Planet-C gameplay video<br>• Research computer vision vs ArUco marker approaches<br>• Record initial chess game footage for testing |
| 2 | Jun 30 - Jul 6 | Conference | ESUG presentation & networking | • Prepare 4-minute project presentation for ESUG<br>• Investigate ArUco tag detection methods<br>• Collect feedback from conference participants<br>• Draft initial project roadmap |
| 3 | Jul 7 - Jul 13 | Planning | Post-conference implementation | • Meet with Christophe to discuss Planet-C specifications<br>• Define MVP: Planet-C detection with live streaming<br>• Plan frame sampling experiments (5th/10th/30th frame)<br>• Set up basic experiment tracking system |
| 4 | Jul 14 - Jul 20 | Infrastructure | Streaming & model setup | • Evaluate RTMP vs wired streaming solutions<br>• Finalize target games: Chess, Planet-C, Ubuntu<br>• Begin Planet-C frame annotation using Roboflow<br>• Start Docker containerization for Planet-C pipeline |
| 5 | Jul 21 - Jul 27 | Development | Annotation pipeline | • Complete 150 manual Planet-C annotations<br>• Experiment with Grounding DINO for auto-labeling<br>• Implement YOLO11 with oriented bounding boxes<br>• Develop board detection using Canny edge detection |
| 6 | Jul 28 - Aug 3 | Training | First model iteration | • Build pseudo-labeling pipeline with board-aware filtering<br>• Train YOLO11-OBB with mixed human/pseudo dataset<br>• Implement standardized logging and folder structure<br>• Design three-bucket annotation workflow |
| 7-9 | Aug 4 - Aug 25 | Break | Holiday period | • Personal time - visiting family in USA<br>• No development work planned<br>• Reflect on project progress and next steps |
| 10 | Aug 25 - Aug 31 | Documentation | Project restart | • Create comprehensive README for YOLO folder<br>• Organize training vs annotation datasets<br>• Implement device-agnostic USB streaming<br>• Begin webinar presentation preparation |
| 11 | Sep 1 - Sep 7 | Model Refinement | Iteration improvements | • Run additional pseudo-labeling cycles<br>• Optimize confidence thresholds and filtering<br>• Implement MLflow for experiment tracking<br>• Test model performance on validation set |
| 12 | Sep 8 - Sep 14 | Integration | CORMAS communication | • Explore Python-Pharo communication methods<br>• Design data exchange protocols<br>• Test live streaming integration<br>• Document API specifications |
| 13 | Sep 15 - Sep 21 | Video Processing | Mentor video analysis | • Extract frames from mentor gameplay video<br>• Run inference on extracted frames<br>• Create visualization of detection results<br>• Prepare analysis for mentor meeting |
| 14 | Sep 22 - Sep 28 | Performance | Optimization & benchmarking | • Profile model inference speed<br>• Optimize for real-time performance<br>• Test on different hardware configurations<br>• Document performance characteristics |
| 15 | Sep 29 - Oct 5 | Webinar | Presentation preparation | • Finalize webinar slides and demo<br>• Prepare live demonstration setup<br>• Test presentation flow and timing<br>• Create backup materials |
| 16 | Oct 6 - Oct 12 | Documentation | Technical documentation | • Create comprehensive technical documentation<br>• Document methodology and results<br>• Prepare figures and performance charts<br>• Organize project artifacts and code |
| 17 | Oct 13 - Oct 19 | Testing | Multi-game validation | • Test Planet-C detection pipeline thoroughly<br>• Validate Ubuntu game detection<br>• Cross-validate Planet-C improvements<br>• Document detection accuracy metrics |
| 18 | Oct 20 - Oct 26 | Deployment | Production readiness | • Package models for distribution<br>• Create installation and setup guides<br>• Test deployment on clean systems<br>• Prepare model release artifacts |
| 19 | Oct 27 - Nov 2 | Integration | CORMAS platform testing | • Test full integration with CORMAS<br>• Validate simulation data exchange<br>• Debug integration issues<br>• Document integration workflow |
| 20 | Nov 3 - Nov 9 | Preparation | ETH trial preparation | • Prepare Planet-C pipeline for ETH demonstration<br>• Test pipeline on different hardware configurations<br>• Create portable deployment package<br>• Prepare demonstration materials and backup plans |
| 21 | Nov 10 - Nov 16 | Pre-deployment | Final preparations | • Conduct final testing and validation<br>• Package all deployment materials<br>• Create backup systems and contingency plans<br>• Finalize travel arrangements for ETH |
| 22 | Nov 17 - Nov 23 | Trial | ETH deployment | • Travel to ETH for pipeline trial<br>• Deploy and test Planet-C detection system<br>• Gather feedback and performance data<br>• Document trial results and lessons learned |