import pygame

screen_width = 1280 # 가로
screen_height = 720 # 세로
VELOCITY = 4
MASS = 2

# 스테이지
stage = pygame.image.load("images/stage.png")
stage_size = stage.get_rect().size
stage_height = stage_size[1] #스테이지의 높이 위에 캐릭터를 두기 위해 사용

objects = []                # 오브젝트 리스트
enemys = []                 # 적 오브젝트 리스트

main_players = pygame.image.load('images/메인남캐_Move_01.gif')
main_players__size = main_players.get_rect().size #이미지의 크기를 구해옴
main_players_width = main_players__size[0] 
main_players_height = main_players__size[1] 

TILE_MAPSIZE = (int(screen_width / 7.5), int(screen_height / 20))
class Player(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y):
        super().__init__()
        
        # Move sprites
        self.sprites = []
        self.Rsprites = []
        
        self.move_direction = True
        self.is_animating = False
        self.is_attack = False
        self.isJump = 0
        
        
        self.sprites.append(pygame.image.load('images/메인남캐_Move_01.gif'))
        self.sprites.append(pygame.image.load('images/메인남캐_Move_02.gif'))
        self.sprites.append(pygame.image.load('images/메인남캐_Move_03.gif'))
        self.sprites.append(pygame.image.load('images/메인남캐_Move_04.gif'))
        self.sprites.append(pygame.image.load('images/메인남캐_Move_05.gif'))
        self.sprites.append(pygame.image.load('images/메인남캐_Move_06.gif'))
        
        self.current_sprite = 0
        self.current_attack_sprite = 0
        
        self.image = self.sprites[self.current_sprite]
        
        self.rect = self.image.get_rect()
        self.rect.y = pos_y
        # self.rect.bottom = 
        # self.rect.topleft = [pos_x, pos_y]
        
        #----------------------------------------------------------------
        ## 이미지 스케일링 ( 사이즈 조절 )
        # for idx,x in enumerate(self.sprites):
        #     x = pygame.transform.scale(x, (200,100))
        #     self.sprites[idx] = x
        
        # 오른쪽 이미지
        for i in (self.sprites):
            self.Rsprites.append(pygame.transform.flip(i, True, False))    

        # 속도
        self.velocity_x = 0
        self.velocity_y = 0
        
        #--------------------------------------------------------
        
        # Attack sprites
        self.attack_sprites = []
        self.attack_Rsprites = []
        
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_01.gif'))
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_02.gif'))
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_03.gif'))
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_04.gif'))
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_05.gif'))
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_06.gif'))
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_07.gif'))
        self.attack_sprites.append(pygame.image.load('images/메인남캐_Attack_08.gif'))
        
        for i in (self.attack_sprites):
            self.attack_Rsprites.append(pygame.transform.flip(i, True, False))  
        
        
        #----------------------------------------------------------------
        # 점프 높이 한계
        self.v = VELOCITY # 속도
        self.m = MASS  # 질량
        
        
    def animate(self):
        self.is_animating = True
        self.velocity_x = 8
        self.velocity_y = 6
        
    def unanimate(self):
        self.is_animating = False
        self.is_attack = False
        self.is_jump = False
        
        self.velocity_x = 0
        
    def jump(self, j):
        self.isJump = j
        
    def update(self):
        
        # 좌우 이동
        if self.is_animating == True: 
            if self.move_direction == True:    # 오른쪽
               self.image = self.Rsprites[int(self.current_sprite)]
               
            else:       #왼쪽
                self.image = self.sprites[int(self.current_sprite)]
                self.velocity_x = abs(self.velocity_x ) * -1
            
            self.current_sprite += 0.2
            
            if self.current_sprite >= len(self.sprites):
                self.current_sprite = 0
                
            self.rect.x += self.velocity_x
        
        # 공격
        if self.is_attack == True:
            if self.move_direction == True:
                self.image = self.attack_Rsprites[int(self.current_attack_sprite)]

            else :
                self.image = self.attack_sprites[int(self.current_attack_sprite)]
                
            self.current_attack_sprite += 0.3
                
            if self.current_attack_sprite >= len(self.attack_sprites):
                self.current_attack_sprite = 0
                self.unanimate()
            
            self.velocity_x = 0
            
        # 점프 
        if self.isJump > 0:
            
        # 점프 한 상태에서 다시 점프를 위한 값
        # 이 코드를 주석처리하면 이중점프를 못한다.
        # if self.isJump == 2:
        #     self.v = VELOCITY

            if self.v > 0:
                # 속도가 0보다 클때는 위로 올라감
                F = (0.5 * self.m * (self.v * self.v))
            else:
                # 속도가 0보다 작을때는 아래로 내려감
                F = -(0.5 * self.m * (self.v * self.v))

        # 좌표 수정 : 위로 올라가기 위해서는 y 좌표를 줄여준다.
            self.rect.y -= round(F)

            # 속도 줄여줌
            self.v -= 1
            
            # 바닥에 닿았을때, 변수 리셋
            if self.rect.bottom > screen_height - stage_height :
                self.rect.bottom = screen_height - stage_height
                self.isJump = 0
                self.v = VELOCITY
                

class Enemy(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y):
        super().__init__()
        
        # Move sprites
        self.enemies = []
        self.Renemies = []
        
        self.enemy = pygame.image.load("images/bird.png")
        
        self.enemy_hpm = 100
        self.enemy_hp = self.enemy_hpm
        self.enemy_size = self.enemy.get_rect().size #이미지의 크기를 구해옴
        self.enemy_width = self.enemy_size[0] # 캐릭터 가로 크기
        self.enemy_height = self.enemy_size[1] # 캐릭터 세로 크기
        self.enemy_alive = True
        self.enemy_x_pos = pos_x
        self.enemy_y_pos = pos_y
        
        self.enemy_rect = self.enemy.get_rect()
        self.enemy_rect.left = self.enemy_x_pos
        self.enemy_rect.top = self.enemy_y_pos
        
        
        self.enemies.append(pygame.image.load('images/메인남캐_Move_01.gif'))
        self.enemies.append(pygame.image.load('images/메인남캐_Move_02.gif'))
        self.enemies.append(pygame.image.load('images/메인남캐_Move_03.gif'))
        self.enemies.append(pygame.image.load('images/메인남캐_Move_04.gif'))
        self.enemies.append(pygame.image.load('images/메인남캐_Move_05.gif'))
        self.enemies.append(pygame.image.load('images/메인남캐_Move_06.gif'))
        
        self.current_enemy = 0
        
        self.image = self.enemies[self.current_enemy]
        
        self.rect = self.image.get_rect()
        self.rect.y = pos_y
        
        # 오른쪽 이미지
        for i in (self.enemies):
            self.Renemies.append(pygame.transform.flip(i, True, False))  
            
        
            
                
#---------------------------------------------------------------------------------#
class BaseObject:
    def __init__(self, spr, coord, kinds, game):
        self.kinds = kinds
        self.spr = spr
        self.spr_index = 0
        self.game = game
        self.width = spr[0].get_width()
        self.height = spr[0].get_height()
        self.move_direction = True
        self.vspeed = 0
        self.gravity = 0.2
        self.movement = [0, 0]
        self.collision = {'top' : False, 'bottom' : False, 'right' : False, 'left' : False}
        self.rect = pygame.rect.Rect(coord[0], coord[1], self.width, self.height)
        self.frameSpeed = 0
        self.frameTimer = 0
        self.destroy = False

    def physics(self):
        self.movement[0] = 0
        self.movement[1] = 0

        if self.gravity != 0:
            self.movement[1] += self.vspeed

            self.vspeed += self.gravity
            if self.vspeed > 3:
                self.vspeed = 3

    # def physics_after(self):
    #     self.rect, self.collision = move(self.rect, self.movement)

    #     if self.collision['bottom']:
    #         self.vspeed = 0

    #     if self.rect.y > 400 or self.rect.y  > 400 or self.rect.y  > 400:
    #         self.destroy = True
    
    def draw(self):
        self.game.screen_scaled.blit(pygame.transform.flip(self.spr[self.spr_index], self.move_direction, False)
                    , (self.rect.x - self.game.camera_scroll[0], self.rect.y - self.game.camera_scroll[1]))

        # if self.kinds == 'enemy' and self.hp < self.hpm:
        #     pygame.draw.rect(self.game.screen_scaled, (131, 133, 131)
        #     , [self.rect.x - 1 - self.game.camera_scroll[0], self.rect.y - 5 - self.game.camera_scroll[1], 10, 2])
        #     pygame.draw.rect(self.game.screen_scaled, (189, 76, 49)
        #     , [self.rect.x - 1 - self.game.camera_scroll[0], self.rect.y - 5 - self.game.camera_scroll[1], 10 * self.hp / self.hpm, 2])

    def animation(self, mode):
        if mode == 'loop':
            self.frameTimer += 1

            if self.frameTimer >= self.frameSpeed:
                self.frameTimer = 0
                if self.spr_index < len(self.spr) - 1:
                    self.spr_index += 1
                else:
                    self.spr_index = 0

    def destroy_self(self):
        if self.kinds == 'enemy':
            enemys.remove(self)

        objects.remove(self)
        del(self)