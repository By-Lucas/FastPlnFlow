from celery import Celery

celery = Celery(
    __name__,
    broker="amqp://fastplnflow:fastplnflow2025@rabbitmq:5672//",
    backend="rpc://"
)

# Descobre tasks em app.tasks
celery.autodiscover_tasks(['app'])
