const onReady = (callback) => {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', callback, { once: true });
  } else {
    callback();
  }
};

const createFormData = (message, files, modelMode) => {
  const formData = new FormData();
  formData.append('message', message);
  formData.append('model_mode', modelMode);
  files.forEach((file) => formData.append('files', file));
  return formData;
};

export default function initChat() {
  onReady(() => {
    const chatForm = document.getElementById('chat-form');
    const chatLog = document.getElementById('chat-log');
    const chatMessage = document.getElementById('chat-message');
    const chatFile = document.getElementById('chat-file');
    const fileList = document.getElementById('file-list');
    const modelMode = document.getElementById('model-mode');
    const modelStatus = document.getElementById('model-status');
    const resetMemoryBtn = document.getElementById('reset-memory');

    if (!chatForm || !chatLog || !chatMessage) {
      return;
    }

    const state = {
      files: [],
    };

    const formatStatusText = (payload) => {
      const parts = [];
      if (payload && payload.model_used) {
        parts.push(`Используется: ${payload.model_used}`);
      }
      if (payload && typeof payload.memory_turns === 'number' && payload.memory_turns > 0) {
        parts.push(`Память: ${payload.memory_turns} ходов`);
      }
      return parts.join(' · ');
    };

    const scrollToBottom = () => {
      requestAnimationFrame(() => {
        chatLog.scrollTop = chatLog.scrollHeight;
      });
    };

    const appendBubble = (role, text, className = '') => {
      const bubble = document.createElement('div');
      bubble.className = `bubble ${role === 'user' ? 'user' : 'bot'}${className ? ` ${className}` : ''}`;
      bubble.textContent = text;
      chatLog.appendChild(bubble);
      scrollToBottom();
      return bubble;
    };

    const appendInfo = (text) => {
      const info = document.createElement('div');
      info.className = 'info-message';
      info.textContent = text;
      chatLog.appendChild(info);
      scrollToBottom();
      return info;
    };

    const updateModelStatus = (statusText) => {
      if (!modelStatus) {
        return;
      }
      modelStatus.textContent = statusText || '';
      modelStatus.className = statusText ? 'model-status active' : 'model-status';
    };

    const renderFiles = () => {
      if (!fileList) {
        return;
      }

      fileList.textContent = '';
      if (state.files.length === 0) {
        return;
      }

      const fragment = document.createDocumentFragment();
      state.files.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'file-item';

        const name = document.createElement('span');
        name.className = 'file-name';
        name.title = file.name;
        name.textContent = file.name;

        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'file-remove';
        remove.dataset.index = String(index);
        remove.setAttribute('aria-label', `Удалить файл ${file.name}`);
        remove.textContent = '×';

        item.append(name, remove);
        fragment.appendChild(item);
      });

      fileList.appendChild(fragment);
    };

    if (fileList) {
      fileList.addEventListener('click', (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
          return;
        }
        if (target.matches('.file-remove')) {
          const index = Number.parseInt(target.dataset.index || '', 10);
          if (!Number.isNaN(index)) {
            state.files.splice(index, 1);
            renderFiles();
          }
        }
      });
    }

    if (chatFile) {
      chatFile.addEventListener('change', (event) => {
        const { files } = event.target;
        if (!files) {
          return;
        }

        Array.from(files).forEach((file) => {
          const isSupported =
            file.type === 'application/pdf' ||
            file.type === 'text/plain' ||
            file.name.toLowerCase().endsWith('.pdf') ||
            file.name.toLowerCase().endsWith('.txt');

          if (!isSupported) {
            return;
          }

          const alreadyAdded = state.files.some(
            (existing) => existing.name === file.name && existing.size === file.size,
          );
          if (!alreadyAdded) {
            state.files.push(file);
          }
        });

        renderFiles();
        event.target.value = '';
      });
    }

    chatForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const message = chatMessage.value.trim();

      if (!message && state.files.length === 0) {
        return;
      }

      if (message) {
        appendBubble('user', message);
      }

      if (state.files.length > 0) {
        const fileNames = state.files.map((file) => file.name).join(', ');
        appendBubble('user', `📎 Файлы: ${fileNames}`);
      }

      chatMessage.value = '';
      const pendingBubble = appendBubble('bot', 'Думаю…');

      const mode = modelMode ? modelMode.value : 'auto';
      updateModelStatus('Обработка...');

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          body: createFormData(message, state.files, mode),
          credentials: 'same-origin',
        });

        const payload = await response.json();

        if (payload.ok && Array.isArray(payload.uploaded_files) && payload.uploaded_files.length) {
          appendInfo(`✓ Файлы "${payload.uploaded_files.join(', ')}" проанализированы и добавлены в базу знаний.`);
        }

        pendingBubble.textContent = payload.ok ? payload.answer : payload.error || 'Ошибка';

        updateModelStatus(formatStatusText(payload));

        if (Array.isArray(payload.snippets) && payload.snippets.length) {
          const fragment = document.createDocumentFragment();
          payload.snippets.forEach((snippet) => {
            const sn = document.createElement('div');
            sn.className = 'snippet';
            sn.textContent = `[${snippet.doc_id || 'doc'}] ${snippet.preview}`;
            fragment.appendChild(sn);
          });
          chatLog.appendChild(fragment);
          scrollToBottom();
        }

        if (payload.ok) {
          state.files.length = 0;
          renderFiles();
        }
      } catch (error) {
        pendingBubble.textContent = `Ошибка: ${String(error)}`;
        updateModelStatus('');
      }
    });

    if (resetMemoryBtn) {
      resetMemoryBtn.addEventListener('click', async () => {
        const originalText = resetMemoryBtn.textContent;
        resetMemoryBtn.disabled = true;
        resetMemoryBtn.textContent = '…';
        try {
          const response = await fetch('/api/chat/reset', {
            method: 'POST',
            credentials: 'same-origin',
          });
          const payload = await response.json();
          if (payload.ok) {
            appendInfo(payload.message || 'Память очищена.');
            updateModelStatus(formatStatusText(payload));
          } else {
            appendInfo(payload.error || 'Не удалось очистить память.');
          }
        } catch (error) {
          appendInfo(`Ошибка очистки памяти: ${String(error)}`);
        } finally {
          resetMemoryBtn.disabled = false;
          resetMemoryBtn.textContent = originalText;
        }
      });
    }
    });
}
