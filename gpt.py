from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

users = []


class User(BaseModel):
    user_name: str
    is_active: bool = False
    id: Optional[int] = None


class UserResponse(BaseModel):
    user_name: str
    is_active: bool


@app.get("/")
async def read_root():
    return {"message": "Hello from app2"}


# @app.get("/user", response_model=List[UserResponse])
# async def get_users():
#     return users


@app.post("/user", response_model=User)
async def create_new_user(user: User):
    new_id = len(users) + 1
    user.id = new_id
    users.append(user)
    return user


@app.get("/user/{id}", response_model=UserResponse)
async def get_user(id: int):
    for user in users:
        if user.id == id:
            return user
    raise HTTPException(status_code=404, detail="User not found")


@app.put("/user/{id}", response_model=UserResponse)
async def update_user(id: int, new_user: User):
    for index, user in enumerate(users):
        if user.id == id:
            users[index] = new_user
            return new_user
    raise HTTPException(status_code=404, detail="User not found")


@app.delete("/user/{id}")
async def delete_user(id: int):
    for index, user in enumerate(users):
        if user.id == id:
            del users[index]
            return {"message": "User deleted successfully"}
    raise HTTPException(status_code=404, detail="User not found, could not be deleted")
