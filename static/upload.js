const onReady = (callback) => {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', callback, { once: true });
  } else {
    callback();
  }
};

export default function initUpload() {
  onReady(() => {
    const uploadForm = document.getElementById('upload-form');
    const uploadResult = document.getElementById('upload-result');

    if (!uploadForm || !uploadResult) {
      return;
    }

    uploadForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      uploadResult.textContent = 'Загрузка…';

      try {
        const formData = new FormData(uploadForm);
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });

        const payload = await response.json();
        uploadResult.textContent = JSON.stringify(payload, null, 2);
      } catch (error) {
        uploadResult.textContent = `Ошибка: ${String(error)}`;
      }
    });
  });
}
