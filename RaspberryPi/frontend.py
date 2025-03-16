<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emoting</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f4f7fa;
            text-align: center;
        }

        h1 {
            font-size: 36px;
            margin-bottom: 20px;
            color: #333;
        }

        .content {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            width: 90%; /* ???볤쾶 */
            max-width: 1400px; /* 理쒕? ?볦씠 異붽? */
        }

        #video-container {
            flex: 1;
            margin-right: 20px;
            display: flex;
            justify-content: center;
        }

        #video-container img {
            width: 640px;
            height: 480px;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        #emotion-info {
            /*flex: 1;  ?띿뒪??諛뺤뒪瑜????볤쾶 */
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            height: fit-content;
            width: 400px;
        }

        #final-score {
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            color: #333;
        }

        .emotion {
            margin-top: 15px;
            font-size: 18px;
            color: #555;
        }

        .emotion span {
            font-weight: bold;
            color: #333;
        }

        .emotion p {
            margin: 5px 0;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>Emoting</h1>
        <div class="content">
            <!-- Video Section -->
            <div id="video-container">
                <img src="{{ url_for('emoting') }}" width="640" height="480" />
            </div>

            <!-- Emotion Information Section -->
            <div id="emotion-info">
                <h3>Facial Expression</h3>
                <div class="emotion">
                    <p><span>Happy:</span> <span id="happy-count">0</span></p>
                    <p><span>Sad:</span> <span id="sad-count">0</span></p>
                    <p><span>Neutral:</span> <span id="neutral-count">0</span></p>
                    <p><span>Disgust:</span> <span id="disgust-count">0</span></p>
                    <p><span>Surprise:</span> <span id="surprise-count">0</span></p>
                    <p><span>Angry:</span> <span id="angry-count">0</span></p>
                </div>

                <h3>Pose</h3>
                <div class="emotion">
                    <p><span>Crossed Arm:</span> <span id="crossed-arm-count">0</span></p>
                    <p><span>Crossed Leg:</span> <span id="crossed-leg-count">0</span></p>
                    <p><span>Smile Cover:</span> <span id="smile-cover-count">0</span></p>
                    <p><span>Touching Chin:</span> <span id="touching-chin-count">0</span></p>
                    <p><span>Hair Comb:</span> <span id="hair-comb-count">0</span></p>
                    <p><span>Drinking:</span> <span id="drinking-count">0</span></p>
                    <p><span>Eating:</span> <span id="eating-count">0</span></p>
                </div>

                <div id="final-score">Final Score: 0</div>
            </div>
        </div>
    </div>

    <script>
        // ?怨쀬뵠????낅쑓??꾨뱜 ??λ땾
        function updateEmotionData() {
            fetch('/update_counts')
                .then(response => response.json())
                .then(data => {
                    // 揶쏅Ŋ??燁삳똻?????낅쑓??꾨뱜
                    document.getElementById('happy-count').textContent = data.expression_counts.happy;
                    document.getElementById('sad-count').textContent = data.expression_counts.sad;
                    document.getElementById('neutral-count').textContent = data.expression_counts.neutral;
                    document.getElementById('disgust-count').textContent = data.expression_counts.disgust;
                    document.getElementById('surprise-count').textContent = data.expression_counts.surprise;
                    document.getElementById('angry-count').textContent = data.expression_counts.angry;

                    // ??已?燁삳똻?????낅쑓??꾨뱜
                    document.getElementById('crossed-arm-count').textContent = data.pose_counts.crossed_arm;
                    document.getElementById('crossed-leg-count').textContent = data.pose_counts.crossed_leg;
                    document.getElementById('smile-cover-count').textContent = data.pose_counts.smile_cover;
                    document.getElementById('touching-chin-count').textContent = data.pose_counts.touching_chin;
                    document.getElementById('hair-comb-count').textContent = data.pose_counts.hair_comb;
                    document.getElementById('drinking-count').textContent = data.pose_counts.drinking;
                    document.getElementById('eating-count').textContent = data.pose_counts.eating;

                    // 筌ㅼ뮇伊??癒?땾 ??낅쑓??꾨뱜
                    document.getElementById('final-score').textContent = 'Final Score: ' + data.final_score;
                });
        }

        // 4?λ뜄彛??揶쏅Ŋ???怨쀬뵠????낅쑓??꾨뱜
        setInterval(updateEmotionData, 4000);
    </script>
</body>
</html>

