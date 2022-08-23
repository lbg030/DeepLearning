import pygame, sys
from datafile import *

class Game:
    def __init__(self): 
        pygame.init()
        pygame.display.set_caption("testing game")

        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((screen_width, screen_height))

        # 캐릭터 움직임
        self.moving_sprites = pygame.sprite.Group()
        self.player = Player(10,screen_height - stage_height - main_players_height)
        self.moving_sprites.add(self.player)

        self.enemy = Enemy(300, screen_height - stage_height - main_players_height)
        
        
        # 게임 실행
        self.run()
        
    # 러닝
    def run(self):
        running = True
        while running :
            
            self.screen.fill((0,0,0))
            
            # 2. 이벤트 처리 ( 키보드, 마우스 등 )
            for event in pygame.event.get(): # 어떤 이벤트가 발생하는지
                if event.type == pygame.QUIT: # 창이 닫히는 이벤트가 발생하면 ( 안쓰면 꺼지지 않음 )
                    running = False
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.player.direction = True
                        self.player.animate()
                        
                    if event.key == pygame.K_LEFT:
                        self.player.direction = False
                        self.player.animate()
                        
                    if event.key == pygame.K_LCTRL:
                        self.player.is_attack = True
                        self.player.animate()
                    
                    if event.key == pygame.K_UP:
                        if self.player.isJump == 0:
                            self.player.jump(1)
                            
                        
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT:
                        self.player.unanimate()

            if self.player.rect.x < 0:
                self.player.rect.x = 0
                
            elif self.player.rect.x > (screen_width - main_players_width):
                self.player.rect.x = screen_width - main_players_width
            
            
            if self.player.rect.colliderect(self.enemy.enemy_rect):

                if self.player.is_attack == True:
                    self.enemy.enemy_hp -= 5
                
            if self.enemy.enemy_hp <= self.enemy.enemy_hpm:
                pygame.draw.rect(self.screen, (131, 133, 131), [(self.enemy.enemy_x_pos), self.enemy.enemy_rect.top - 5 , self.enemy.enemy_width, 40])
                pygame.draw.rect(self.screen, (189, 76, 49), [self.enemy.enemy_x_pos, self.enemy.enemy_rect.top - 5 , self.enemy.enemy_width * self.enemy.enemy_hp / self.enemy.enemy_hpm, 40])

            if self.enemy.enemy_hp < 1:
                self.enemy.enemy_alive = False
                
            if self.enemy.enemy_alive:
                self.screen.blit(self.enemy.enemy, (self.enemy.enemy_x_pos, self.enemy.enemy_y_pos))
            
            
            
            self.screen.blit(stage, (0, screen_height - stage_height))
            self.moving_sprites.draw(self.screen)
            
            self.moving_sprites.update()
            
            
            pygame.display.update() #게임 화면을 다시 그리기 ! (반드시 계속 호출 되어야 되는 부분)
            self.clock.tick(60) #게임 화면의 초당 프레임 수를 설정
        
    # # 파이게임 종료
    # pygame.quit()
    
game = Game()   # 게임 실행