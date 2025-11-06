// Upload form handler
document.addEventListener('DOMContentLoaded', () => {
  const uploadForm = document.getElementById('upload-form');
  const uploadResult = document.getElementById('upload-result');
  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      uploadResult.textContent = 'Загрузка…';
      const formData = new FormData(uploadForm);
      try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        uploadResult.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        uploadResult.textContent = 'Ошибка: ' + String(err);
      }
    });
  }

  // Chat handler
  const chatForm = document.getElementById('chat-form');
  const chatLog = document.getElementById('chat-log');
  const chatMsg = document.getElementById('chat-message');
  const chatFile = document.getElementById('chat-file');
  const fileList = document.getElementById('file-list');
  const modelMode = document.getElementById('model-mode');
  const modelStatus = document.getElementById('model-status');
  
  // Хранилище выбранных файлов
  let selectedFiles = [];
  
  // Обработчик выбора файлов
  if (chatFile) {
    chatFile.addEventListener('change', (e) => {
      const files = Array.from(e.target.files);
      files.forEach(file => {
        if (file.type === 'application/pdf' || file.type === 'text/plain' || 
            file.name.endsWith('.pdf') || file.name.endsWith('.txt')) {
          if (!selectedFiles.find(f => f.name === file.name && f.size === file.size)) {
            selectedFiles.push(file);
            updateFileList();
          }
        }
      });
      // Сбрасываем input, чтобы можно было выбрать тот же файл снова
      e.target.value = '';
    });
  }
  
  // Обновление списка файлов в UI
  function updateFileList() {
    if (!fileList) return;
    fileList.innerHTML = '';
    selectedFiles.forEach((file, index) => {
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = `
        <span class="file-name" title="${file.name}">${file.name}</span>
        <span class="file-remove" data-index="${index}">×</span>
      `;
      fileList.appendChild(item);
    });
    
    // Обработчики удаления файлов
    fileList.querySelectorAll('.file-remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(e.target.getAttribute('data-index'));
        selectedFiles.splice(index, 1);
        updateFileList();
      });
    });
  }
  
  if (chatForm && chatLog && chatMsg) {
    const append = (role, text, className = '') => {
      const div = document.createElement('div');
      div.className = 'bubble ' + (role === 'user' ? 'user' : 'bot') + (className ? ' ' + className : '');
      div.textContent = text;
      chatLog.appendChild(div);
      chatLog.scrollTop = chatLog.scrollHeight;
      return div;
    };
    
    const appendInfo = (text) => {
      const div = document.createElement('div');
      div.className = 'info-message';
      div.textContent = text;
      chatLog.appendChild(div);
      chatLog.scrollTop = chatLog.scrollHeight;
      return div;
    };

    // Обновление статуса модели
    const updateModelStatus = (status) => {
      if (modelStatus) {
        modelStatus.textContent = status || '';
        modelStatus.className = status ? 'model-status active' : 'model-status';
      }
    };

    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = chatMsg.value.trim();
      if (!message && selectedFiles.length === 0) return;
      
      if (message) {
        append('user', message);
      }
      
      // Показываем информацию о загруженных файлах
      if (selectedFiles.length > 0) {
        const fileNames = selectedFiles.map(f => f.name).join(', ');
        append('user', `📎 Файлы: ${fileNames}`);
      }
      
      chatMsg.value = '';
      const pending = append('bot', 'Думаю…');
      
      // Получаем выбранный режим модели
      const selectedMode = modelMode ? modelMode.value : 'auto';
      updateModelStatus('Обработка...');
      
      try {
        // Создаем FormData для отправки файлов
        const formData = new FormData();
        formData.append('message', message || '');
        formData.append('model_mode', selectedMode);
        
        // Добавляем файлы
        selectedFiles.forEach((file, index) => {
          formData.append('files', file);
        });
        
        const resp = await fetch('/api/chat', {
          method: 'POST',
          body: formData
        });
        const data = await resp.json();
        
        // Показываем информацию о загруженных файлах перед ответом
        if (data.ok && data.uploaded_files && data.uploaded_files.length > 0) {
          const fileNames = data.uploaded_files.join(', ');
          appendInfo(`✓ Файлы "${fileNames}" проанализированы и добавлены в базу знаний.`);
        }
        
        pending.textContent = data.ok ? data.answer : (data.error || 'Ошибка');
        
        // Обновляем статус модели из ответа сервера
        if (data.model_used) {
          updateModelStatus(`Используется: ${data.model_used}`);
        } else {
          updateModelStatus('');
        }
        
        if (data.snippets && data.snippets.length) {
          data.snippets.forEach(s => {
            const sn = document.createElement('div');
            sn.className = 'snippet';
            sn.textContent = `[${s.doc_id || 'doc'}] ${s.preview}`;
            chatLog.appendChild(sn);
          });
        }
        
        // Очищаем список файлов после успешной отправки
        if (data.ok) {
          selectedFiles = [];
          updateFileList();
        }
      } catch (err) {
        pending.textContent = 'Ошибка: ' + String(err);
        updateModelStatus('');
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    });
  }
});


