const pageId = document.body?.dataset?.page || '';

const loaders = {
  upload: () => import('./upload.js'),
  chat: () => import('./chat.js'),
};

const bootstrap = loaders[pageId];

if (bootstrap) {
  bootstrap()
    .then(({ default: init }) => {
      if (typeof init === 'function') {
        init();
      }
    })
    .catch((error) => {
      console.error('Не удалось инициализировать сценарий страницы', error);
    });
}