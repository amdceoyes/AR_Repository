import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """通用的带进度条的下载函数"""
    print(f"[INFO] 开始下载: {filename} ...")
    
    # 发送网络请求，允许重定向（GitHub 资源需要）
    response = requests.get(url, stream=True, allow_redirects=True, verify=False)
    total_size = int(response.headers.get('content-length', 0))
    
    # 检查网络请求是否成功
    if response.status_code != 200:
        print(f"[ERROR] 下载失败！HTTP 状态码: {response.status_code}")
        print("提示: 可能是网络连接 GitHub 较慢，请稍后重试或尝试挂起代理。")
        return False

    # 写入本地物理文件，并显示进度条
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
        progress_bar.close()
    
    # 只要本地实实在在写入了数据，并且没有网络中断，就认为成功
    if progress_bar.n > 0:
        print(f"[SUCCESS] {filename} 下载完成并已安全存盘！\n")
        return True
    else:
        print("[ERROR] 下载数据为空，请检查网络。")
        return False

if __name__ == "__main__":
    # 两个官方预训练模型的真实直接下载链接
    urls = {
        "deploy.prototxt": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt",
        "mobilenet_iter_73000.caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    }
    
    print("=======================================================")
    
    print("      FranceHonor AI 节点 - 本地小模型弹药装填")
    print("=======================================================\n")
    
    all_success = True
    for filename, url in urls.items():
        # 如果本地已经有了，就不重复下载，防止刷流量
        if os.path.exists(filename):
            print(f"[INFO] 侦测到本地已存在 {filename}，跳过下载。")
            continue
            
        success = download_file(url, filename)
        if not success:
            all_success = False
            break
            
    if all_success:
        print("🎉 [恭喜] 所有 AI 模型文件已就位！你可以去启动三端大总装了！")