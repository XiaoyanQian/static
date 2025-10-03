"""
conda create --yes --name service_asr_client python==3.10.*
conda activate service_asr_client
pip install websockets sounddevice numpy
"""

import json
import asyncio
import websockets
import numpy as np
import sounddevice as sd
from queue import Queue
from websockets.exceptions import ConnectionClosed


class ASRClient:
    def __init__(self, uri, duration):
        self.uri = uri
        self.duration = duration

        self.queue = Queue()
        self.stream = None
        self.connect = None
        self.last_data = None
        self.is_running = False
        self.waiting_for_stop = False

    def record(self):
        '''启动录音并将分块的录音压入队列'''
        def receive(indata, *args):
            pcm = (indata * 32767).astype(np.int16).tobytes()
            self.queue.put(pcm)

        sample_rate = 16000  # 16k HZ
        blocksize = int(sample_rate * (self.duration / 1000))
        self.stream = sd.InputStream(
            dtype=np.float32,
            blocksize=blocksize,
            channels=1,
            samplerate=sample_rate,
            callback=receive
        )
        self.stream.start()

    async def send(self):
        '''周期性50ms路由队列是否为空,将路由发送给服务器'''
        while self.is_running:
            try:
                if not self.queue.empty():  # 删除(修改)
                    pcm = self.queue.get()
                    await self.connect.send(pcm)
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"Send audio record error:{e}")
                await self.stop()
                break

    async def receive(self):
        while self.is_running:
            try:
                data = json.loads(await self.connect.recv())
                if data.get('type') == 'ready_to_stop':
                    self.waiting_for_stop = False  # 表示等待已结束

                    if self.last_data:
                        data = self.last_data
                        print_result(
                            True,
                            data.get('lines', []),
                            data.get('buffer_transcription', ''),
                        )
                    await self.connect.close()
                    break
                else:
                    self.last_data = data
                    print_result(
                        False,
                        data.get('lines', []),
                        data.get('buffer_transcription', ''),
                        data.get('remaining_time_transcription', 0),
                        data.get('status', 'active_transcription')
                    )
            except ConnectionClosed:
                print("Disconnected from the client.")
                if self.waiting_for_stop and self.last_data:
                    data = self.last_data
                    print_result(
                        True,
                        data.get('lines', []),
                        data.get('buffer_transcription', ''),
                    )
                else:
                    print("Disconnected from the WebSocket server.")
                if self.is_running:
                    await self.stop()
                break

    async def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.connect:
            self.waiting_for_stop = True
            await self.connect.close()
            self.connect = None
        self.is_running = False

    async def start(self):
        try:
            # 1. 创建到ASR服务器的连接
            self.connect = await websockets.connect(self.uri)

            # 2. 启动麦克风录音
            self.record()

            # 3. 设置Client为运行状态
            self.is_running = True

            # 4. 启动发送录音块的协程循环
            # 5. 启动接受服务器解雇的携程循环
            asyncio.create_task(self.send())
            asyncio.create_task(self.receive())
        except Exception as e:
            print(f'Starting error:{e}')
            await self.stop()


def print_result(stopped, lines, buffer_transcription="", remaining_time_transcription=0, status="active_transcription"):
    if status != "no_audio_detected":
        for line in lines:
            time_info = ""

            beg, end = line.get('beg'), line.get('end')
            if beg and end:
                time_info = f" {beg} - {end}"

            speaker_label = ""
            speaker = line.get('speaker')
            if speaker == -2:
                speaker_label = f"[Silence]{time_info}"
            elif speaker == -1:
                speaker_label = f"[Speaker 1]{time_info}"
            else:
                speaker_label = f"[Speaker {speaker}]{time_info}"
            text = line.get('text', '')
            if text or speaker_label:
                print(f"\n{speaker_label}")
                if text:
                    print(f"\t{text}")

        if buffer_transcription:
            if stopped:
                print(f"\t{buffer_transcription}")
            else:
                print(f"\t[Buffer Transcription] {buffer_transcription}")
        if not stopped and remaining_time_transcription > 0:
            print(f"\t[Transcription lag: {remaining_time_transcription}s]")
        print('\n\n')


async def main():
    asrClient = ASRClient("ws://localhost:55000/asr", 2000)
    while True:
        prompt = "Enter your choice (1-3): "
        print("1. Start recording")
        print("2. Stop recording")
        print("3. Exit")

        choice = (await asyncio.to_thread(input, prompt)).strip()
        if choice == "1":
            if asrClient.is_running:
                print("Is running.\n")
                continue
            else:
                await asrClient.start()
        elif choice == "2":
            if asrClient.is_running:
                await asrClient.stop()
            else:
                print("Not running!\n")
        elif choice == "3":
            if asrClient.is_running:
                await asrClient.stop()
            break
        else:
            print("Invalid choice.\n")

if __name__ == "__main__":
    asyncio.run(main())
