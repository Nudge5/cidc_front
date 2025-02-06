const chatContainer = document.getElementById('chat-container'); // chat-container ID 기준으로 Element 가져오기
const queryInput = document.getElementById('query-input'); // query-input ID 기준으로 Element 가져오기

async function askQuestion() {
    const query = queryInput.value; // 사용자의 질문 내용을 가져옴
    if (!query) return; // 질문이 비어있으면 함수 종료

    addMessage('Me', query, 'user'); // 사용자 메시지를 추가
    queryInput.value = ''; // 입력창 초기화

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }), // 사용자의 질문 데이터를 POST 요청으로 보냄
            credentials: 'same-origin'
        });

        const data = await response.json(); // 서버 응답을 JSON으로 파싱
        addMessage('Bot', data.answer, 'bot'); // 봇의 응답을 메시지로 추가
    } catch (error) {
        console.error('Error in askQuestion:', error);
        addMessage('Error', `Failed to get answer: ${error.message}`, 'error'); // 에러 메시지를 추가
    }
}

function addMessage(sender, message, className) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className}`;

    // 아바타 이미지 추가
    const avatar = document.createElement('img');
    avatar.className = 'avatar';
    if (className === 'user') {
        avatar.src = '/static/me.png'; // 사용자 이미지 경로
        avatar.alt = 'User Avatar';
    } else if (className === 'bot') {
        avatar.src = '/static/bot.png'; // 봇 이미지 경로
        avatar.alt = 'Bot Avatar';
    } else {
        avatar.src = ''; // 에러나 기본 상태 이미지
        avatar.alt = 'Default Avatar';
    }

    // 텍스트 요소 추가
    const textElement = document.createElement('div');
    textElement.className = 'text';
    if (className === 'bot') {
        // 봇의 응답을 마크다운 형식으로 처리
        textElement.innerHTML = `<strong>${sender}:</strong> ${marked.parse(message)}`;
    } else {
        textElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    }

    // 메시지 요소 구성
    messageElement.appendChild(avatar); // 이미지 추가
    messageElement.appendChild(textElement); // 텍스트 추가

    // 채팅 컨테이너에 메시지 추가
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight; // 스크롤을 최신 메시지로 이동
}

queryInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        askQuestion(); // Enter 키 입력 시 질문 함수 실행
    }
});
