# SPDX-License-Identifier: MIT-0
from threading import Thread
from typing import Callable
from awsiot.greengrasscoreipc import connect
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    IoTCoreMessage,
    QOS,
    PublishToIoTCoreRequest,
    SubscribeToIoTCoreRequest,
    UpdateStateRequest
)
import concurrent.futures
import logging


TIMEOUT = 10

ipc_client = connect()

def sync_error_handler(error: Exception) -> None:
    raise error

class SubscribeHandler(client.SubscribeToIoTCoreStreamHandler):
    def __init__(self, handler: Callable[[str, bytes], None], error_handler: Callable[[Exception], None] ):
        self._handler = handler
        self._error_handler = error_handler

    def on_stream_event(self, event: IoTCoreMessage) -> None:
        msg = event.message
        t = Thread(target=self._handler, args=[msg.topic_name, msg.payload])
        t.start()

    def on_stream_error(self, error: Exception)-> bool:
        t = Thread(target=self._error_handler, args=[error])
        t.start()
        return True

    def on_stream_closed(self) -> None:
        pass

def publish_async(topic: str, message: bytes, qos: QOS) -> concurrent.futures.Future:
    request = PublishToIoTCoreRequest()
    request.topic_name = topic
    request.payload = message
    request.qos = qos
    operation = ipc_client.new_publish_to_iot_core()
    operation.activate(request)
    future = operation.get_response()
    return future

def publish(topic: str, message: bytes, qos: QOS):
    try:
        future = publish_async(topic, message, qos)
        future.result(TIMEOUT)
    except Exception as ex:
        raise ex

def subscribe_async(topic: str, qos: QOS, handler: Callable[[str, bytes], None], error_handler: Callable[[Exception], None]) -> concurrent.futures.Future:
    request = SubscribeToIoTCoreRequest()
    request.topic_name = topic
    request.qos = qos
    handler = SubscribeHandler(handler, error_handler)
    operation = ipc_client.new_subscribe_to_iot_core(handler)
    operation.activate(request)
    future = operation.get_response()
    return future



def subscribe(topic: str, qos: QOS, handler: Callable[[str, bytes], None]):
    try:
        future = subscribe_async(topic, qos, handler, sync_error_handler)
        future.result(TIMEOUT)
    except Exception as ex:
        raise ex


def set_running():
    state = UpdateStateRequest(state="RUNNING")
    op = ipc_client.new_update_state()
    res = op.activate(state)
    res.result()