import rl.connection as connection

conn = connection.connect_socket_connection("172.18.237.16", 8889)
#%%

cmd, data = connection.send_recv(conn, ("model", "latest"))
cmd, data = connection.send_recv(conn, ("model", -1))
cmd, data = connection.send_recv(conn, ("model", "sample_opponent"))
cmd, data = connection.send_recv(conn, ("model", "eval_opponent"))
#%%

cmd, data = connection.send_recv(conn, ("eval_infos", [{"model_id": ("test", 12),
                                                         "opponent_id": ("test", 3),
                                                         "win": 1,
                                                         "reward": 1}]
                                        )
                                 )
